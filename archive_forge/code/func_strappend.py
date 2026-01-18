import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
def strappend(self, name: str, value: str, path: Optional[str]=Path.root_path()) -> Union[int, List[Optional[int]]]:
    """Append to the string JSON value. If two options are specified after
        the key name, the path is determined to be the first. If a single
        option is passed, then the root_path (i.e Path.root_path()) is used.

        For more information see `JSON.STRAPPEND <https://redis.io/commands/json.strappend>`_.
        """
    pieces = [name, str(path), self._encode(value)]
    return self.execute_command('JSON.STRAPPEND', *pieces)