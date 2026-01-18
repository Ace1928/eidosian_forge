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
@resolves(pydantic.ConstrainedBytes)
def resolve_conbytes(cls):
    min_size = cls.min_length or 0
    max_size = cls.max_length
    if not cls.strip_whitespace:
        return st.binary(min_size=min_size, max_size=max_size)
    repeats = '{{{},{}}}'.format(min_size - 2 if min_size > 2 else 0, max_size - 2 if (max_size or 0) > 2 else '')
    if min_size >= 2:
        pattern = f'\\W.{repeats}\\W'
    elif min_size == 1:
        pattern = f'\\W(.{repeats}\\W)?'
    else:
        assert min_size == 0
        pattern = f'(\\W(.{repeats}\\W)?)?'
    return st.from_regex(pattern.encode(), fullmatch=True)