from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def extend_arg(arg: GraphQLArgument) -> GraphQLArgument:
    return GraphQLArgument(**merge_kwargs(arg.to_kwargs(), type_=replace_type(arg.type)))