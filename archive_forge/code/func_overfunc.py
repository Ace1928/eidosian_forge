from __future__ import annotations
import codecs
import re
import textwrap
from collections.abc import Hashable, Mapping
from functools import reduce
from operator import or_ as set_union
from re import Pattern
from typing import TYPE_CHECKING, Any, Callable, Generic
from unicodedata import normalize
import numpy as np
from xarray.core import duck_array_ops
from xarray.core.computation import apply_ufunc
from xarray.core.types import T_DataArray
def overfunc(x, iwidth, ifillchar):
    if len(ifillchar) != 1:
        raise TypeError('fillchar must be a character, not str')
    return func(x, int(iwidth), ifillchar)