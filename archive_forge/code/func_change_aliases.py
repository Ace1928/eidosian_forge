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
def change_aliases(self, change_map):
    """
        Change the aliases in change_map (which maps old-alias -> new-alias),
        relabelling any references to them in select columns and the where
        clause.
        """
    assert set(change_map).isdisjoint(change_map.values())
    self.where.relabel_aliases(change_map)
    if isinstance(self.group_by, tuple):
        self.group_by = tuple([col.relabeled_clone(change_map) for col in self.group_by])
    self.select = tuple([col.relabeled_clone(change_map) for col in self.select])
    self.annotations = self.annotations and {key: col.relabeled_clone(change_map) for key, col in self.annotations.items()}
    for old_alias, new_alias in change_map.items():
        if old_alias not in self.alias_map:
            continue
        alias_data = self.alias_map[old_alias].relabeled_clone(change_map)
        self.alias_map[new_alias] = alias_data
        self.alias_refcount[new_alias] = self.alias_refcount[old_alias]
        del self.alias_refcount[old_alias]
        del self.alias_map[old_alias]
        table_aliases = self.table_map[alias_data.table_name]
        for pos, alias in enumerate(table_aliases):
            if alias == old_alias:
                table_aliases[pos] = new_alias
                break
    self.external_aliases = {change_map.get(alias, alias): aliased or alias in change_map for alias, aliased in self.external_aliases.items()}