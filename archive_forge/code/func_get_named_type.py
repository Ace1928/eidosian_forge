from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
def get_named_type(type):
    unmodified_type = type
    while isinstance(unmodified_type, (GraphQLList, GraphQLNonNull)):
        unmodified_type = unmodified_type.of_type
    return unmodified_type