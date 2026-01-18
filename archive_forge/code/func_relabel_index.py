from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
import re
from typing import (
from uuid import uuid4
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCSeries
from pandas import (
from pandas.api.types import is_list_like
import pandas.core.common as com
from markupsafe import escape as escape_html  # markupsafe is jinja2 dependency
def relabel_index(self, labels: Sequence | Index, axis: Axis=0, level: Level | list[Level] | None=None) -> StylerRenderer:
    """
        Relabel the index, or column header, keys to display a set of specified values.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        labels : list-like or Index
            New labels to display. Must have same length as the underlying values not
            hidden.
        axis : {"index", 0, "columns", 1}
            Apply to the index or columns.
        level : int, str, list, optional
            The level(s) over which to apply the new labels. If `None` will apply
            to all levels of an Index or MultiIndex which are not hidden.

        Returns
        -------
        Styler

        See Also
        --------
        Styler.format_index: Format the text display value of index or column headers.
        Styler.hide: Hide the index, column headers, or specified data from display.

        Notes
        -----
        As part of Styler, this method allows the display of an index to be
        completely user-specified without affecting the underlying DataFrame data,
        index, or column headers. This means that the flexibility of indexing is
        maintained whilst the final display is customisable.

        Since Styler is designed to be progressively constructed with method chaining,
        this method is adapted to react to the **currently specified hidden elements**.
        This is useful because it means one does not have to specify all the new
        labels if the majority of an index, or column headers, have already been hidden.
        The following produce equivalent display (note the length of ``labels`` in
        each case).

        .. code-block:: python

            # relabel first, then hide
            df = pd.DataFrame({"col": ["a", "b", "c"]})
            df.style.relabel_index(["A", "B", "C"]).hide([0,1])
            # hide first, then relabel
            df = pd.DataFrame({"col": ["a", "b", "c"]})
            df.style.hide([0,1]).relabel_index(["C"])

        This method should be used, rather than :meth:`Styler.format_index`, in one of
        the following cases (see examples):

          - A specified set of labels are required which are not a function of the
            underlying index keys.
          - The function of the underlying index keys requires a counter variable,
            such as those available upon enumeration.

        Examples
        --------
        Basic use

        >>> df = pd.DataFrame({"col": ["a", "b", "c"]})
        >>> df.style.relabel_index(["A", "B", "C"])  # doctest: +SKIP
             col
        A      a
        B      b
        C      c

        Chaining with pre-hidden elements

        >>> df.style.hide([0,1]).relabel_index(["C"])  # doctest: +SKIP
             col
        C      c

        Using a MultiIndex

        >>> midx = pd.MultiIndex.from_product([[0, 1], [0, 1], [0, 1]])
        >>> df = pd.DataFrame({"col": list(range(8))}, index=midx)
        >>> styler = df.style  # doctest: +SKIP
                  col
        0  0  0     0
              1     1
           1  0     2
              1     3
        1  0  0     4
              1     5
           1  0     6
              1     7
        >>> styler.hide((midx.get_level_values(0)==0)|(midx.get_level_values(1)==0))
        ...  # doctest: +SKIP
        >>> styler.hide(level=[0,1])  # doctest: +SKIP
        >>> styler.relabel_index(["binary6", "binary7"])  # doctest: +SKIP
                  col
        binary6     6
        binary7     7

        We can also achieve the above by indexing first and then re-labeling

        >>> styler = df.loc[[(1,1,0), (1,1,1)]].style
        >>> styler.hide(level=[0,1]).relabel_index(["binary6", "binary7"])
        ...  # doctest: +SKIP
                  col
        binary6     6
        binary7     7

        Defining a formatting function which uses an enumeration counter. Also note
        that the value of the index key is passed in the case of string labels so it
        can also be inserted into the label, using curly brackets (or double curly
        brackets if the string if pre-formatted),

        >>> df = pd.DataFrame({"samples": np.random.rand(10)})
        >>> styler = df.loc[np.random.randint(0,10,3)].style
        >>> styler.relabel_index([f"sample{i+1} ({{}})" for i in range(3)])
        ...  # doctest: +SKIP
                         samples
        sample1 (5)     0.315811
        sample2 (0)     0.495941
        sample3 (2)     0.067946
        """
    axis = self.data._get_axis_number(axis)
    if axis == 0:
        display_funcs_, obj = (self._display_funcs_index, self.index)
        hidden_labels, hidden_lvls = (self.hidden_rows, self.hide_index_)
    else:
        display_funcs_, obj = (self._display_funcs_columns, self.columns)
        hidden_labels, hidden_lvls = (self.hidden_columns, self.hide_columns_)
    visible_len = len(obj) - len(set(hidden_labels))
    if len(labels) != visible_len:
        raise ValueError(f'``labels`` must be of length equal to the number of visible labels along ``axis`` ({visible_len}).')
    if level is None:
        level = [i for i in range(obj.nlevels) if not hidden_lvls[i]]
    levels_ = refactor_levels(level, obj)

    def alias_(x, value):
        if isinstance(value, str):
            return value.format(x)
        return value
    for ai, i in enumerate([i for i in range(len(obj)) if i not in hidden_labels]):
        if len(levels_) == 1:
            idx = (i, levels_[0]) if axis == 0 else (levels_[0], i)
            display_funcs_[idx] = partial(alias_, value=labels[ai])
        else:
            for aj, lvl in enumerate(levels_):
                idx = (i, lvl) if axis == 0 else (lvl, i)
                display_funcs_[idx] = partial(alias_, value=labels[ai][aj])
    return self