import copy
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, TypeVar, Union
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.convert import as_type
from triad.utils.iter import to_kv_iterable
def set_value_by_index(self, index: int, value: VT) -> None:
    """Set value by index

        :param index: index of the item
        :param value: new value
        """
    key = self.get_key_by_index(index)
    self[key] = value