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
@Substitution(subset=subset_args, props=properties_args, color=coloring_args.format(default='red'))
def highlight_null(self, color: str='red', subset: Subset | None=None, props: str | None=None) -> Styler:
    """
        Highlight missing values with a style.

        Parameters
        ----------
        %(color)s

            .. versionadded:: 1.5.0

        %(subset)s

        %(props)s

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler

        See Also
        --------
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, np.nan]})
        >>> df.style.highlight_null(color='yellow')  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """

    def f(data: DataFrame, props: str) -> np.ndarray:
        return np.where(pd.isna(data).to_numpy(), props, '')
    if props is None:
        props = f'background-color: {color};'
    return self.apply(f, axis=None, subset=subset, props=props)