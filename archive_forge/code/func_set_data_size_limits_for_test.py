from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
def set_data_size_limits_for_test(max_total_size: int, max_resource_size: int) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    For testing.

    **EXPERIMENTAL**

    :param max_total_size: Maximum total buffer size.
    :param max_resource_size: Maximum per-resource size.
    """
    params: T_JSON_DICT = dict()
    params['maxTotalSize'] = max_total_size
    params['maxResourceSize'] = max_resource_size
    cmd_dict: T_JSON_DICT = {'method': 'Network.setDataSizeLimitsForTest', 'params': params}
    json = (yield cmd_dict)