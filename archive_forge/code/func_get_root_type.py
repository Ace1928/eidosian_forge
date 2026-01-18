from copy import copy, deepcopy
from typing import (
from ..error import GraphQLError
from ..language import ast, OperationType
from ..pyutils import inspect, is_collection, is_description
from .definition import (
from .directives import GraphQLDirective, specified_directives, is_directive
from .introspection import introspection_types
def get_root_type(self, operation: OperationType) -> Optional[GraphQLObjectType]:
    return getattr(self, f'{operation.value}_type')