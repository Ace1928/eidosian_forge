from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def is_type_factory(_type: type[Any]) -> Callable[[Any], None]:
    """

    Parameters
    ----------
    `_type` - a type to be compared against (e.g. type(x) == `_type`)

    Returns
    -------
    validator - a function of a single argument x , which raises
                ValueError if type(x) is not equal to `_type`

    """

    def inner(x) -> None:
        if type(x) != _type:
            raise ValueError(f"Value must have type '{_type}'")
    return inner