from __future__ import annotations
import inspect
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping import aes
def get_kwarg_names(func):
    """
    Return a list of valid kwargs to function func
    """
    sig = inspect.signature(func)
    kwonlyargs = [p.name for p in sig.parameters.values() if p.default is not p.empty]
    return kwonlyargs