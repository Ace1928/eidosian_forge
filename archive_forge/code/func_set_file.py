import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
def set_file(self, name: str, path: str, file_name: str, nx: Optional[bool]=False, xx: Optional[bool]=False, decode_keys: Optional[bool]=False) -> Optional[str]:
    """
        Set the JSON value at key ``name`` under the ``path`` to the content
        of the json file ``file_name``.

        ``nx`` if set to True, set ``value`` only if it does not exist.
        ``xx`` if set to True, set ``value`` only if it exists.
        ``decode_keys`` If set to True, the keys of ``obj`` will be decoded
        with utf-8.

        """
    with open(file_name, 'r') as fp:
        file_content = loads(fp.read())
    return self.set(name, path, file_content, nx=nx, xx=xx, decode_keys=decode_keys)