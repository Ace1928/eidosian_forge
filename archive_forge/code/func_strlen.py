import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
def strlen(self, name: str, path: Optional[str]=None) -> List[Union[int, None]]:
    """Return the length of the string JSON value under ``path`` at key
        ``name``.

        For more information see `JSON.STRLEN <https://redis.io/commands/json.strlen>`_.
        """
    pieces = [name]
    if path is not None:
        pieces.append(str(path))
    return self.execute_command('JSON.STRLEN', *pieces)