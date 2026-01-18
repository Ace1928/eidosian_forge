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
@Substitution(subset=subset_args, color=coloring_args.format(default='yellow'), props=properties_args)
def highlight_min(self, subset: Subset | None=None, color: str='yellow', axis: Axis | None=0, props: str | None=None) -> Styler:
    """
        Highlight the minimum with a style.

        Parameters
        ----------
        %(subset)s
        %(color)s
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(props)s

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [2, 1], 'B': [3, 4]})
        >>> df.style.highlight_min(color='yellow')  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
    if props is None:
        props = f'background-color: {color};'
    return self.apply(partial(_highlight_value, op='min'), axis=axis, subset=subset, props=props)