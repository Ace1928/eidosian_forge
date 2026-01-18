import copy
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, TypeVar, Union
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.convert import as_type
from triad.utils.iter import to_kv_iterable
def index_of_key(self, key: Any) -> int:
    """Get index of key

        :param key: key value
        :return: index of the key value
        """
    self._build_index()
    return self._key_index[key]