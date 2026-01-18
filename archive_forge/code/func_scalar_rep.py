from __future__ import annotations
import functools
import re
import textwrap
from typing import (
import unicodedata
import numpy as np
from pandas._libs import lib
import pandas._libs.missing as libmissing
import pandas._libs.ops as libops
from pandas.core.dtypes.missing import isna
from pandas.core.strings.base import BaseStringArrayMethods
def scalar_rep(x):
    try:
        return bytes.__mul__(x, rint)
    except TypeError:
        return str.__mul__(x, rint)