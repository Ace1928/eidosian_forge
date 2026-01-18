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
@resolves(pydantic.ConstrainedInt)
def resolve_conint(cls):
    min_value = cls.ge
    max_value = cls.le
    if cls.gt is not None:
        assert min_value is None, 'Set `gt` or `ge`, but not both'
        min_value = cls.gt + 1
    if cls.lt is not None:
        assert max_value is None, 'Set `lt` or `le`, but not both'
        max_value = cls.lt - 1
    if cls.multiple_of is None or cls.multiple_of == 1:
        return st.integers(min_value, max_value)
    if min_value is not None:
        min_value = math.ceil(Fraction(min_value) / Fraction(cls.multiple_of))
    if max_value is not None:
        max_value = math.floor(Fraction(max_value) / Fraction(cls.multiple_of))
    return st.integers(min_value, max_value).map(lambda x: x * cls.multiple_of)