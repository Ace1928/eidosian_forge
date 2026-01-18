import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
def mset(self, triplets: List[Tuple[str, str, JsonType]]) -> Optional[str]:
    """
        Set the JSON value at key ``name`` under the ``path`` to ``obj``
        for one or more keys.

        ``triplets`` is a list of one or more triplets of key, path, value.

        For the purpose of using this within a pipeline, this command is also
        aliased to JSON.MSET.

        For more information see `JSON.MSET <https://redis.io/commands/json.mset>`_.
        """
    pieces = []
    for triplet in triplets:
        pieces.extend([triplet[0], str(triplet[1]), self._encode(triplet[2])])
    return self.execute_command('JSON.MSET', *pieces)