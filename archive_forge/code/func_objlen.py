import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
def objlen(self, name: str, path: Optional[str]=Path.root_path()) -> List[Optional[int]]:
    """Return the length of the dictionary JSON value under ``path`` at key
        ``name``.

        For more information see `JSON.OBJLEN <https://redis.io/commands/json.objlen>`_.
        """
    return self.execute_command('JSON.OBJLEN', name, str(path))