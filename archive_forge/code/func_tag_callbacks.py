import collections
from enum import Enum
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
def tag_callbacks(callbacks: Optional[List['Callback']]) -> bool:
    """Records built-in callback usage via a JSON str representing a
    dictionary mapping callback class name -> counts.

    User-defined callbacks will increment the count under the `CustomLoggerCallback`
    or `CustomCallback` key depending on which of the provided interfaces they subclass.
    NOTE: This will NOT track the name of the user-defined callback,
    nor its implementation.

    This will NOT report telemetry if no callbacks are provided by the user.

    Returns:
        bool: True if usage was recorded, False otherwise.
    """
    if not callbacks:
        return False
    callback_counts = _count_callbacks(callbacks)
    if callback_counts:
        callback_counts_str = json.dumps(callback_counts)
        record_extra_usage_tag(TagKey.AIR_CALLBACKS, callback_counts_str)