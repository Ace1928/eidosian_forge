from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_async_call_stack_depth(max_depth: int) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enables or disables async call stacks tracking.

    :param max_depth: Maximum depth of async call stacks. Setting to ```0``` will effectively disable collecting async call stacks (default).
    """
    params: T_JSON_DICT = dict()
    params['maxDepth'] = max_depth
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.setAsyncCallStackDepth', 'params': params}
    json = (yield cmd_dict)