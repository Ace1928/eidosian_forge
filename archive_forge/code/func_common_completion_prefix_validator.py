from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def common_completion_prefix_validator(df: pd.DataFrame) -> Remediation:
    """
    This validator will suggest to remove a common prefix from the completion if a long one exist.
    """
    MAX_PREFIX_LEN = 5
    common_prefix = get_common_xfix(df.completion, xfix='prefix')
    ws_prefix = len(common_prefix) > 0 and common_prefix[0] == ' '
    if len(common_prefix) < MAX_PREFIX_LEN:
        return Remediation(name='common_prefix')

    def remove_common_prefix(x: Any, prefix: Any, ws_prefix: Any) -> Any:
        x['completion'] = x['completion'].str[len(prefix):]
        if ws_prefix:
            x['completion'] = f' {x['completion']}'
        return x
    if (df.completion == common_prefix).all():
        return Remediation(name='common_prefix')
    immediate_msg = f'\n- All completions start with prefix `{common_prefix}`. Most of the time you should only add the output data into the completion, without any prefix'
    optional_msg = f'Remove prefix `{common_prefix}` from all completions'

    def optional_fn(x: Any) -> Any:
        return remove_common_prefix(x, common_prefix, ws_prefix)
    return Remediation(name='common_completion_prefix', immediate_msg=immediate_msg, optional_msg=optional_msg, optional_fn=optional_fn)