from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
def label_from_attrs(da: DataArray | None, extra: str='') -> str:
    """Makes informative labels if variable metadata (attrs) follows
    CF conventions."""
    if da is None:
        return ''
    name: str = '{}'
    if 'long_name' in da.attrs:
        name = name.format(da.attrs['long_name'])
    elif 'standard_name' in da.attrs:
        name = name.format(da.attrs['standard_name'])
    elif da.name is not None:
        name = name.format(da.name)
    else:
        name = ''
    units = _get_units_from_attrs(da)
    if name.startswith('$') and name.count('$') % 2 == 0:
        return '$\n$'.join(textwrap.wrap(name + extra + units, 60, break_long_words=False))
    else:
        return '\n'.join(textwrap.wrap(name + extra + units, 30))