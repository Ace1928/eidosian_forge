from typing import Any, List, Optional, Union
import numpy as np
def adjust_high(low: float, high: float, q: float, include_high: bool):
    _high = low + np.floor((high - low) / q + _IGNORABLE_ERROR) * q
    if abs(_high - high) < _IGNORABLE_ERROR:
        if include_high:
            _high = high + q
    else:
        _high += q
    return _high