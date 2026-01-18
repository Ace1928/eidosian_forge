from typing import Any, Callable, NamedTuple, Optional, Union
from graphql_relay.utils.base64 import base64, unbase64
from graphql import (
def from_global_id(global_id: str) -> ResolvedGlobalId:
    """
    Takes the "global ID" created by to_global_id, and returns the type name and ID
    used to create it.
    """
    global_id = unbase64(global_id)
    if ':' not in global_id:
        return ResolvedGlobalId('', global_id)
    return ResolvedGlobalId(*global_id.split(':', 1))