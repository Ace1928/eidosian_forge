import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
@deprecated_function(version='4.0.0', reason='redisjson-py supported this, call get directly.')
def jsonmget(self, *args, **kwargs):
    return self.mget(*args, **kwargs)