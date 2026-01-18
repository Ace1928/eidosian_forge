from typing import Union, Iterable, Sequence, Any, Optional, Iterator
import sys
import json as _builtin_json
import gzip
from . import ujson
from .util import force_path, force_string, FilePath, JSONInput, JSONOutput
def json_loads(data: Union[str, bytes]) -> JSONOutput:
    """Deserialize unicode or bytes to a Python object.

    data (str / bytes): The data to deserialize.
    RETURNS: The deserialized Python object.
    """
    if data == '-':
        raise ValueError('Expected object or value')
    return ujson.loads(data)