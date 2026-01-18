from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
def parse_literal(self, value_ast):
    if isinstance(value_ast, ast.EnumValue):
        enum_value = self._name_lookup.get(value_ast.value)
        if enum_value:
            return enum_value.value