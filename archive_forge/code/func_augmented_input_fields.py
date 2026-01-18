from collections.abc import Mapping
from inspect import iscoroutinefunction
from typing import Any, Callable, Dict, Optional
from graphql import (
from graphql.pyutils import AwaitableOrValue
def augmented_input_fields() -> GraphQLInputFieldMap:
    return dict(resolve_thunk(input_fields), clientMutationId=GraphQLInputField(GraphQLString))