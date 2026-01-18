from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
def get_nullable_type(type):
    if isinstance(type, GraphQLNonNull):
        return type.of_type
    return type