from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def replace_directive(directive: GraphQLDirective) -> GraphQLDirective:
    kwargs = directive.to_kwargs()
    return GraphQLDirective(**merge_kwargs(kwargs, args={name: extend_arg(arg) for name, arg in kwargs['args'].items()}))