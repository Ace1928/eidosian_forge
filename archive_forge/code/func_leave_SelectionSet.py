import itertools
from collections import OrderedDict
from ...error import GraphQLError
from ...language import ast
from ...language.printer import print_ast
from ...pyutils.pair_set import PairSet
from ...type.definition import (GraphQLInterfaceType, GraphQLList,
from ...utils.type_comparators import is_equal_type
from ...utils.type_from_ast import type_from_ast
from .base import ValidationRule
def leave_SelectionSet(self, node, key, parent, path, ancestors):
    conflicts = _find_conflicts_within_selection_set(self.context, self._cached_fields_and_fragment_names, self._compared_fragments, self.context.get_parent_type(), node)
    for (reason_name, reason), fields1, fields2 in conflicts:
        self.context.report_error(GraphQLError(self.fields_conflict_message(reason_name, reason), list(fields1) + list(fields2)))