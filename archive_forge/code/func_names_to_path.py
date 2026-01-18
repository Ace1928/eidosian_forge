import copy
import difflib
import functools
import sys
from collections import Counter, namedtuple
from collections.abc import Iterator, Mapping
from itertools import chain, count, product
from string import ascii_uppercase
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import DEFAULT_DB_ALIAS, NotSupportedError, connections
from django.db.models.aggregates import Count
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import (
from django.db.models.fields import Field
from django.db.models.fields.related_lookups import MultiColSource
from django.db.models.lookups import Lookup
from django.db.models.query_utils import (
from django.db.models.sql.constants import INNER, LOUTER, ORDER_DIR, SINGLE
from django.db.models.sql.datastructures import BaseTable, Empty, Join, MultiJoin
from django.db.models.sql.where import AND, OR, ExtraWhere, NothingNode, WhereNode
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
from django.utils.tree import Node
def names_to_path(self, names, opts, allow_many=True, fail_on_missing=False):
    """
        Walk the list of names and turns them into PathInfo tuples. A single
        name in 'names' can generate multiple PathInfos (m2m, for example).

        'names' is the path of names to travel, 'opts' is the model Options we
        start the name resolving from, 'allow_many' is as for setup_joins().
        If fail_on_missing is set to True, then a name that can't be resolved
        will generate a FieldError.

        Return a list of PathInfo tuples. In addition return the final field
        (the last used join field) and target (which is a field guaranteed to
        contain the same value as the final field). Finally, return those names
        that weren't found (which are likely transforms and the final lookup).
        """
    path, names_with_path = ([], [])
    for pos, name in enumerate(names):
        cur_names_with_path = (name, [])
        if name == 'pk' and opts is not None:
            name = opts.pk.name
        field = None
        filtered_relation = None
        try:
            if opts is None:
                raise FieldDoesNotExist
            field = opts.get_field(name)
        except FieldDoesNotExist:
            if name in self.annotation_select:
                field = self.annotation_select[name].output_field
            elif name in self._filtered_relations and pos == 0:
                filtered_relation = self._filtered_relations[name]
                if LOOKUP_SEP in filtered_relation.relation_name:
                    parts = filtered_relation.relation_name.split(LOOKUP_SEP)
                    filtered_relation_path, field, _, _ = self.names_to_path(parts, opts, allow_many, fail_on_missing)
                    path.extend(filtered_relation_path[:-1])
                else:
                    field = opts.get_field(filtered_relation.relation_name)
        if field is not None:
            if field.is_relation and (not field.related_model):
                raise FieldError('Field %r does not generate an automatic reverse relation and therefore cannot be used for reverse querying. If it is a GenericForeignKey, consider adding a GenericRelation.' % name)
            try:
                model = field.model._meta.concrete_model
            except AttributeError:
                model = None
        else:
            pos -= 1
            if pos == -1 or fail_on_missing:
                available = sorted([*get_field_names_from_opts(opts), *self.annotation_select, *self._filtered_relations])
                raise FieldError("Cannot resolve keyword '%s' into field. Choices are: %s" % (name, ', '.join(available)))
            break
        if opts is not None and model is not opts.model:
            path_to_parent = opts.get_path_to_parent(model)
            if path_to_parent:
                path.extend(path_to_parent)
                cur_names_with_path[1].extend(path_to_parent)
                opts = path_to_parent[-1].to_opts
        if hasattr(field, 'path_infos'):
            if filtered_relation:
                pathinfos = field.get_path_info(filtered_relation)
            else:
                pathinfos = field.path_infos
            if not allow_many:
                for inner_pos, p in enumerate(pathinfos):
                    if p.m2m:
                        cur_names_with_path[1].extend(pathinfos[0:inner_pos + 1])
                        names_with_path.append(cur_names_with_path)
                        raise MultiJoin(pos + 1, names_with_path)
            last = pathinfos[-1]
            path.extend(pathinfos)
            final_field = last.join_field
            opts = last.to_opts
            targets = last.target_fields
            cur_names_with_path[1].extend(pathinfos)
            names_with_path.append(cur_names_with_path)
        else:
            final_field = field
            targets = (field,)
            if fail_on_missing and pos + 1 != len(names):
                raise FieldError("Cannot resolve keyword %r into field. Join on '%s' not permitted." % (names[pos + 1], name))
            break
    return (path, final_field, targets, names[pos + 1:])