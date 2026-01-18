import operator
from functools import reduce
from django.core.exceptions import EmptyResultSet, FullResultSet
from django.db.models.expressions import Case, When
from django.db.models.functions import Mod
from django.db.models.lookups import Exact
from django.utils import tree
from django.utils.functional import cached_property
def split_having_qualify(self, negated=False, must_group_by=False):
    """
        Return three possibly None nodes: one for those parts of self that
        should be included in the WHERE clause, one for those parts of self
        that must be included in the HAVING clause, and one for those parts
        that refer to window functions.
        """
    if not self.contains_aggregate and (not self.contains_over_clause):
        return (self, None, None)
    in_negated = negated ^ self.negated
    must_remain_connected = in_negated and self.connector == AND or (not in_negated and self.connector == OR) or self.connector == XOR
    if must_remain_connected and self.contains_aggregate and (not self.contains_over_clause):
        return (None, self, None)
    where_parts = []
    having_parts = []
    qualify_parts = []
    for c in self.children:
        if hasattr(c, 'split_having_qualify'):
            where_part, having_part, qualify_part = c.split_having_qualify(in_negated, must_group_by)
            if where_part is not None:
                where_parts.append(where_part)
            if having_part is not None:
                having_parts.append(having_part)
            if qualify_part is not None:
                qualify_parts.append(qualify_part)
        elif c.contains_over_clause:
            qualify_parts.append(c)
        elif c.contains_aggregate:
            having_parts.append(c)
        else:
            where_parts.append(c)
    if must_remain_connected and qualify_parts:
        if not where_parts or (where_parts and (not must_group_by)):
            return (None, None, self)
        elif where_parts:
            raise NotImplementedError('Heterogeneous disjunctive predicates against window functions are not implemented when performing conditional aggregation.')
    where_node = self.create(where_parts, self.connector, self.negated) if where_parts else None
    having_node = self.create(having_parts, self.connector, self.negated) if having_parts else None
    qualify_node = self.create(qualify_parts, self.connector, self.negated) if qualify_parts else None
    return (where_node, having_node, qualify_node)