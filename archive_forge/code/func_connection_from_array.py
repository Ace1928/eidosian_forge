from typing import Any, Iterator, Optional, Sequence
from ..utils.base64 import base64, unbase64
from .connection import (
def connection_from_array(data: SizedSliceable, args: Optional[ConnectionArguments]=None, connection_type: ConnectionConstructor=Connection, edge_type: EdgeConstructor=Edge, page_info_type: PageInfoConstructor=PageInfo) -> ConnectionType:
    """Create a connection object from a sequence of objects.

    Note that different from its JavaScript counterpart which expects an array,
    this function accepts any kind of sliceable object with a length.

    Given this `data` object representing the result set, and connection arguments,
    this simple function returns a connection object for use in GraphQL. It uses
    offsets as pagination, so pagination will only work if the data is static.

    The result will use the default types provided in the `connectiontypes` module
    if you don't pass custom types as arguments.
    """
    return connection_from_array_slice(data, args, slice_start=0, array_length=len(data), connection_type=connection_type, edge_type=edge_type, page_info_type=page_info_type)