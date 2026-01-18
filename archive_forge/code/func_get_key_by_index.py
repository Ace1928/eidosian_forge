import copy
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, TypeVar, Union
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.convert import as_type
from triad.utils.iter import to_kv_iterable
def get_key_by_index(self, index: int) -> KT:
    """Get key by index

        :param index: index of the key
        :return: key value at the index
        """
    self._build_index()
    return self._index_key[index]