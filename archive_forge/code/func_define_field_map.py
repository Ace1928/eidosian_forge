from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
def define_field_map(type, field_map):
    if callable(field_map):
        field_map = field_map()
    assert isinstance(field_map, Mapping) and len(field_map) > 0, '{} fields must be a mapping (dict / OrderedDict) with field names as keys or a function which returns such a mapping.'.format(type)
    for field_name, field in field_map.items():
        assert_valid_name(field_name)
        field_args = getattr(field, 'args', None)
        if field_args:
            assert isinstance(field_args, Mapping), '{}.{} args must be a mapping (dict / OrderedDict) with argument names as keys.'.format(type, field_name)
            for arg_name, arg in field_args.items():
                assert_valid_name(arg_name)
    return OrderedDict(field_map)