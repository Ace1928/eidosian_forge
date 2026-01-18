import calendar
import datetime
import datetime as dt
import importlib
import logging
import numbers
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from redis.exceptions import ResponseError
from .exceptions import TimeoutFormatError
def decode_redis_hash(h) -> Dict[str, Any]:
    """Decodes the Redis hash, ensuring that keys are strings
    Most importantly, decodes bytes strings, ensuring the dict has str keys.

    Args:
        h (Dict[Any, Any]): The Redis hash

    Returns:
        Dict[str, Any]: The decoded Redis data (Dictionary)
    """
    return dict(((as_text(k), h[k]) for k in h))