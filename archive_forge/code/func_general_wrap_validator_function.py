from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
@deprecated('`general_wrap_validator_function` is deprecated, use `with_info_wrap_validator_function` instead.')
def general_wrap_validator_function(*args, **kwargs):
    warnings.warn('`general_wrap_validator_function` is deprecated, use `with_info_wrap_validator_function` instead.', DeprecationWarning)
    return with_info_wrap_validator_function(*args, **kwargs)