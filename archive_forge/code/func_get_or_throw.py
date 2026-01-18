import copy
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, TypeVar, Union
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.convert import as_type
from triad.utils.iter import to_kv_iterable
def get_or_throw(self, key: Union[int, str], expected_type: type) -> Any:
    """Get value by ``key``, and the value must be a subtype of ``expected_type``.
        If ``key`` is not found or value can't be converted to ``expected_type``, raise
        exception

        :param key: the key to search
        :param expected_type: expected return value type

        :raises KeyError: if ``key`` is not found
        :raises TypeError: if the value can't be converted to ``expected_type``

        :return: only when ``key`` is found and can be converted to ``expected_type``,
            return the converted value
        """
    return self._get_or(key, expected_type, throw=True)