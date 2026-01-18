import contextlib
import datetime
import ipaddress
import json
import math
from fractions import Fraction
from typing import Callable, Dict, Type, Union, cast, overload
import hypothesis.strategies as st
import pydantic
import pydantic.color
import pydantic.types
from pydantic.utils import lenient_issubclass
@resolves(pydantic.ConstrainedFloat)
def resolve_confloat(cls):
    min_value = cls.ge
    max_value = cls.le
    exclude_min = False
    exclude_max = False
    if cls.gt is not None:
        assert min_value is None, 'Set `gt` or `ge`, but not both'
        min_value = cls.gt
        exclude_min = True
    if cls.lt is not None:
        assert max_value is None, 'Set `lt` or `le`, but not both'
        max_value = cls.lt
        exclude_max = True
    if cls.multiple_of is None:
        return st.floats(min_value, max_value, exclude_min=exclude_min, exclude_max=exclude_max, allow_nan=False)
    if min_value is not None:
        min_value = math.ceil(min_value / cls.multiple_of)
        if exclude_min:
            min_value = min_value + 1
    if max_value is not None:
        assert max_value >= cls.multiple_of, 'Cannot build model with max value smaller than multiple of'
        max_value = math.floor(max_value / cls.multiple_of)
        if exclude_max:
            max_value = max_value - 1
    return st.integers(min_value, max_value).map(lambda x: x * cls.multiple_of)