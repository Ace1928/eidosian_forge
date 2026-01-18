from __future__ import annotations
from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas import (
import pandas.core.common as com
from pandas.core.frame import (
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import (
@doc(apply_index, this='map', wise='elementwise', alt='apply', altwise='level-wise', func='take a scalar and return a string', input_note='an index value, if an Index, or a level value of a MultiIndex', output_note='CSS styles as a string', var='v', ret='"background-color: yellow;" if v == "B" else None', ret2='"background-color: yellow;" if "x" in v else None')
def map_index(self, func: Callable, axis: AxisInt | str=0, level: Level | list[Level] | None=None, **kwargs) -> Styler:
    self._todo.append((lambda instance: getattr(instance, '_apply_index'), (func, axis, level, 'map'), kwargs))
    return self