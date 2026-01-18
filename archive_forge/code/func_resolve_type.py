import functools
from wandb_promise import Promise, is_thenable, promise_for_dict
from ...pyutils.cached_property import cached_property
from ...pyutils.default_ordered_dict import DefaultOrderedDict
from ...type import (GraphQLInterfaceType, GraphQLList, GraphQLNonNull,
from ..base import ResolveInfo, Undefined, collect_fields, get_field_def
from ..values import get_argument_values
from ...error import GraphQLError
def resolve_type(self, result):
    return_type = self.abstract_type
    context = self.context.context_value
    if return_type.resolve_type:
        return return_type.resolve_type(result, context, self.info)
    for type, is_type_of in self.possible_types_with_is_type_of:
        if is_type_of(result, context, self.info):
            return type