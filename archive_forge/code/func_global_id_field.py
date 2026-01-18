from typing import Any, Callable, NamedTuple, Optional, Union
from graphql_relay.utils.base64 import base64, unbase64
from graphql import (
def global_id_field(type_name: Optional[str]=None, id_fetcher: Optional[Callable[[Any, GraphQLResolveInfo], str]]=None) -> GraphQLField:
    """
    Creates the configuration for an id field on a node, using `to_global_id` to
    construct the ID from the provided typename. The type-specific ID is fetched
    by calling id_fetcher on the object, or if not provided, by accessing the `id`
    attribute of the object, or the `id` if the object is a dict.
    """

    def resolve(obj: Any, info: GraphQLResolveInfo, **_args: Any) -> str:
        type_ = type_name or info.parent_type.name
        id_ = id_fetcher(obj, info) if id_fetcher else obj['id'] if isinstance(obj, dict) else obj.id
        return to_global_id(type_, id_)
    return GraphQLField(GraphQLNonNull(GraphQLID), description='The ID of an object', resolve=resolve)