import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
def arrtrim(self, name: str, path: str, start: int, stop: int) -> List[Union[int, None]]:
    """Trim the array JSON value under ``path`` at key ``name`` to the
        inclusive range given by ``start`` and ``stop``.

        For more information see `JSON.ARRTRIM <https://redis.io/commands/json.arrtrim>`_.
        """
    return self.execute_command('JSON.ARRTRIM', name, str(path), start, stop)