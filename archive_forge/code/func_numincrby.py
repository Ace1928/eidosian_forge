import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
def numincrby(self, name: str, path: str, number: int) -> str:
    """Increment the numeric (integer or floating point) JSON value under
        ``path`` at key ``name`` by the provided ``number``.

        For more information see `JSON.NUMINCRBY <https://redis.io/commands/json.numincrby>`_.
        """
    return self.execute_command('JSON.NUMINCRBY', name, str(path), self._encode(number))