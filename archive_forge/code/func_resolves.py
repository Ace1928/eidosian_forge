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
def resolves(typ: Union[type, pydantic.types.ConstrainedNumberMeta]) -> Callable[[Callable[..., st.SearchStrategy]], Callable[..., st.SearchStrategy]]:

    def inner(f):
        assert f not in RESOLVERS
        RESOLVERS[typ] = f
        return f
    return inner