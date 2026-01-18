from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
def is_composite_type(type):
    named_type = get_named_type(type)
    return isinstance(named_type, (GraphQLObjectType, GraphQLInterfaceType, GraphQLUnionType))