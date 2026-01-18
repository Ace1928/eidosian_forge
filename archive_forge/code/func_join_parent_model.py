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
def join_parent_model(self, opts, model, alias, seen):
    """
        Make sure the given 'model' is joined in the query. If 'model' isn't
        a parent of 'opts' or if it is None this method is a no-op.

        The 'alias' is the root alias for starting the join, 'seen' is a dict
        of model -> alias of existing joins. It must also contain a mapping
        of None -> some alias. This will be returned in the no-op case.
        """
    if model in seen:
        return seen[model]
    chain = opts.get_base_chain(model)
    if not chain:
        return alias
    curr_opts = opts
    for int_model in chain:
        if int_model in seen:
            curr_opts = int_model._meta
            alias = seen[int_model]
            continue
        if not curr_opts.parents[int_model]:
            curr_opts = int_model._meta
            continue
        link_field = curr_opts.get_ancestor_link(int_model)
        join_info = self.setup_joins([link_field.name], curr_opts, alias)
        curr_opts = int_model._meta
        alias = seen[int_model] = join_info.joins[-1]
    return alias or seen[None]