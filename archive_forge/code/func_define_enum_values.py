from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
def define_enum_values(type, value_map):
    assert isinstance(value_map, Mapping) and len(value_map) > 0, '{} values must be a mapping (dict / OrderedDict) with value names as keys.'.format(type)
    values = []
    if not isinstance(value_map, (collections.OrderedDict, OrderedDict)):
        value_map = OrderedDict(sorted(list(value_map.items())))
    for value_name, value in value_map.items():
        assert_valid_name(value_name)
        assert isinstance(value, GraphQLEnumValue), '{}.{} must be an instance of GraphQLEnumValue, but got: {}'.format(type, value_name, value)
        value = copy.copy(value)
        value.name = value_name
        if value.value is None:
            value.value = value_name
        values.append(value)
    return values