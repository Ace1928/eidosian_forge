import collections
import json
import re
from functools import partial
from itertools import chain
from django.core.exceptions import EmptyResultSet, FieldError, FullResultSet
from django.db import DatabaseError, NotSupportedError
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import F, OrderBy, RawSQL, Ref, Value
from django.db.models.functions import Cast, Random
from django.db.models.lookups import Lookup
from django.db.models.query_utils import select_related_descend
from django.db.models.sql.constants import (
from django.db.models.sql.query import Query, get_order_dir
from django.db.models.sql.where import AND
from django.db.transaction import TransactionManagementError
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
from django.utils.regex_helper import _lazy_re_compile
def get_related_selections(self, select, select_mask, opts=None, root_alias=None, cur_depth=1, requested=None, restricted=None):
    """
        Fill in the information needed for a select_related query. The current
        depth is measured as the number of connections away from the root model
        (for example, cur_depth=1 means we are looking at models with direct
        connections to the root model).
        """

    def _get_field_choices():
        direct_choices = (f.name for f in opts.fields if f.is_relation)
        reverse_choices = (f.field.related_query_name() for f in opts.related_objects if f.field.unique)
        return chain(direct_choices, reverse_choices, self.query._filtered_relations)
    related_klass_infos = []
    if not restricted and cur_depth > self.query.max_depth:
        return related_klass_infos
    if not opts:
        opts = self.query.get_meta()
        root_alias = self.query.get_initial_alias()
    fields_found = set()
    if requested is None:
        restricted = isinstance(self.query.select_related, dict)
        if restricted:
            requested = self.query.select_related

    def get_related_klass_infos(klass_info, related_klass_infos):
        klass_info['related_klass_infos'] = related_klass_infos
    for f in opts.fields:
        fields_found.add(f.name)
        if restricted:
            next = requested.get(f.name, {})
            if not f.is_relation:
                if next or f.name in requested:
                    raise FieldError("Non-relational field given in select_related: '%s'. Choices are: %s" % (f.name, ', '.join(_get_field_choices()) or '(none)'))
        else:
            next = False
        if not select_related_descend(f, restricted, requested, select_mask):
            continue
        related_select_mask = select_mask.get(f) or {}
        klass_info = {'model': f.remote_field.model, 'field': f, 'reverse': False, 'local_setter': f.set_cached_value, 'remote_setter': f.remote_field.set_cached_value if f.unique else lambda x, y: None, 'from_parent': False}
        related_klass_infos.append(klass_info)
        select_fields = []
        _, _, _, joins, _, _ = self.query.setup_joins([f.name], opts, root_alias)
        alias = joins[-1]
        columns = self.get_default_columns(related_select_mask, start_alias=alias, opts=f.remote_field.model._meta)
        for col in columns:
            select_fields.append(len(select))
            select.append((col, None))
        klass_info['select_fields'] = select_fields
        next_klass_infos = self.get_related_selections(select, related_select_mask, f.remote_field.model._meta, alias, cur_depth + 1, next, restricted)
        get_related_klass_infos(klass_info, next_klass_infos)
    if restricted:
        related_fields = [(o.field, o.related_model) for o in opts.related_objects if o.field.unique and (not o.many_to_many)]
        for related_field, model in related_fields:
            related_select_mask = select_mask.get(related_field) or {}
            if not select_related_descend(related_field, restricted, requested, related_select_mask, reverse=True):
                continue
            related_field_name = related_field.related_query_name()
            fields_found.add(related_field_name)
            join_info = self.query.setup_joins([related_field_name], opts, root_alias)
            alias = join_info.joins[-1]
            from_parent = issubclass(model, opts.model) and model is not opts.model
            klass_info = {'model': model, 'field': related_field, 'reverse': True, 'local_setter': related_field.remote_field.set_cached_value, 'remote_setter': related_field.set_cached_value, 'from_parent': from_parent}
            related_klass_infos.append(klass_info)
            select_fields = []
            columns = self.get_default_columns(related_select_mask, start_alias=alias, opts=model._meta, from_parent=opts.model)
            for col in columns:
                select_fields.append(len(select))
                select.append((col, None))
            klass_info['select_fields'] = select_fields
            next = requested.get(related_field.related_query_name(), {})
            next_klass_infos = self.get_related_selections(select, related_select_mask, model._meta, alias, cur_depth + 1, next, restricted)
            get_related_klass_infos(klass_info, next_klass_infos)

        def local_setter(final_field, obj, from_obj):
            if from_obj:
                final_field.remote_field.set_cached_value(from_obj, obj)

        def local_setter_noop(obj, from_obj):
            pass

        def remote_setter(name, obj, from_obj):
            setattr(from_obj, name, obj)
        for name in list(requested):
            if cur_depth > 1:
                break
            if name in self.query._filtered_relations:
                fields_found.add(name)
                final_field, _, join_opts, joins, _, _ = self.query.setup_joins([name], opts, root_alias)
                model = join_opts.model
                alias = joins[-1]
                from_parent = issubclass(model, opts.model) and model is not opts.model
                klass_info = {'model': model, 'field': final_field, 'reverse': True, 'local_setter': partial(local_setter, final_field) if len(joins) <= 2 else local_setter_noop, 'remote_setter': partial(remote_setter, name), 'from_parent': from_parent}
                related_klass_infos.append(klass_info)
                select_fields = []
                field_select_mask = select_mask.get((name, final_field)) or {}
                columns = self.get_default_columns(field_select_mask, start_alias=alias, opts=model._meta, from_parent=opts.model)
                for col in columns:
                    select_fields.append(len(select))
                    select.append((col, None))
                klass_info['select_fields'] = select_fields
                next_requested = requested.get(name, {})
                next_klass_infos = self.get_related_selections(select, field_select_mask, opts=model._meta, root_alias=alias, cur_depth=cur_depth + 1, requested=next_requested, restricted=restricted)
                get_related_klass_infos(klass_info, next_klass_infos)
        fields_not_found = set(requested).difference(fields_found)
        if fields_not_found:
            invalid_fields = ("'%s'" % s for s in fields_not_found)
            raise FieldError('Invalid field name(s) given in select_related: %s. Choices are: %s' % (', '.join(invalid_fields), ', '.join(_get_field_choices()) or '(none)'))
    return related_klass_infos