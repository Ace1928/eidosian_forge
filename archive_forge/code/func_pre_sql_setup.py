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
def pre_sql_setup(self):
    """
        If the update depends on results from other tables, munge the "where"
        conditions to match the format required for (portable) SQL updates.

        If multiple updates are required, pull out the id values to update at
        this point so that they don't change as a result of the progressive
        updates.
        """
    refcounts_before = self.query.alias_refcount.copy()
    self.query.get_initial_alias()
    count = self.query.count_active_tables()
    if not self.query.related_updates and count == 1:
        return
    query = self.query.chain(klass=Query)
    query.select_related = False
    query.clear_ordering(force=True)
    query.extra = {}
    query.select = []
    meta = query.get_meta()
    fields = [meta.pk.name]
    related_ids_index = []
    for related in self.query.related_updates:
        if all((path.join_field.primary_key for path in meta.get_path_to_parent(related))):
            related_ids_index.append((related, 0))
        else:
            related_ids_index.append((related, len(fields)))
            fields.append(related._meta.pk.name)
    query.add_fields(fields)
    super().pre_sql_setup()
    must_pre_select = count > 1 and (not self.connection.features.update_can_self_select)
    self.query.clear_where()
    if self.query.related_updates or must_pre_select:
        idents = []
        related_ids = collections.defaultdict(list)
        for rows in query.get_compiler(self.using).execute_sql(MULTI):
            idents.extend((r[0] for r in rows))
            for parent, index in related_ids_index:
                related_ids[parent].extend((r[index] for r in rows))
        self.query.add_filter('pk__in', idents)
        self.query.related_ids = related_ids
    else:
        self.query.add_filter('pk__in', query)
    self.query.reset_refcounts(refcounts_before)