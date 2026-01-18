import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
def arrappend(self, name: str, path: Optional[str]=Path.root_path(), *args: List[JsonType]) -> List[Union[int, None]]:
    """Append the objects ``args`` to the array under the
        ``path` in key ``name``.

        For more information see `JSON.ARRAPPEND <https://redis.io/commands/json.arrappend>`_..
        """
    pieces = [name, str(path)]
    for o in args:
        pieces.append(self._encode(o))
    return self.execute_command('JSON.ARRAPPEND', *pieces)