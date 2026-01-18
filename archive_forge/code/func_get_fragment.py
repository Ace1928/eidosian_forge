import functools
from wandb_promise import Promise, is_thenable, promise_for_dict
from ...pyutils.cached_property import cached_property
from ...pyutils.default_ordered_dict import DefaultOrderedDict
from ...type import (GraphQLInterfaceType, GraphQLList, GraphQLNonNull,
from ..base import ResolveInfo, Undefined, collect_fields, get_field_def
from ..values import get_argument_values
from ...error import GraphQLError
def get_fragment(self, type):
    if isinstance(type, str):
        type = self.context.schema.get_type(type)
    if type not in self._fragments:
        assert type in self.possible_types, 'Runtime Object type "{}" is not a possible type for "{}".'.format(type, self.abstract_type)
        self._fragments[type] = Fragment(type, self.field_asts, self.context, self.info)
    return self._fragments[type]