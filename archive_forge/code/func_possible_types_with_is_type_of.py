import functools
from wandb_promise import Promise, is_thenable, promise_for_dict
from ...pyutils.cached_property import cached_property
from ...pyutils.default_ordered_dict import DefaultOrderedDict
from ...type import (GraphQLInterfaceType, GraphQLList, GraphQLNonNull,
from ..base import ResolveInfo, Undefined, collect_fields, get_field_def
from ..values import get_argument_values
from ...error import GraphQLError
@cached_property
def possible_types_with_is_type_of(self):
    return [(type, type.is_type_of) for type in self.possible_types if callable(type.is_type_of)]