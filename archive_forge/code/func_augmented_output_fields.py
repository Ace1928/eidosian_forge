from collections.abc import Mapping
from inspect import iscoroutinefunction
from typing import Any, Callable, Dict, Optional
from graphql import (
from graphql.pyutils import AwaitableOrValue
def augmented_output_fields() -> GraphQLFieldMap:
    return dict(resolve_thunk(output_fields), clientMutationId=GraphQLField(GraphQLString))