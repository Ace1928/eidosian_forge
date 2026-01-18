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
def set_uuid(self, uuid: str) -> Styler:
    """
        Set the uuid applied to ``id`` attributes of HTML elements.

        Parameters
        ----------
        uuid : str

        Returns
        -------
        Styler

        Notes
        -----
        Almost all HTML elements within the table, and including the ``<table>`` element
        are assigned ``id`` attributes. The format is ``T_uuid_<extra>`` where
        ``<extra>`` is typically a more specific identifier, such as ``row1_col2``.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=['A', 'B'], columns=['c1', 'c2'])

        You can get the `id` attributes with the following:

        >>> print((df).style.to_html())  # doctest: +SKIP

        To add a title to column `c1`, its `id` is T_20a7d_level0_col0:

        >>> df.style.set_uuid("T_20a7d_level0_col0")
        ... .set_caption("Test")  # doctest: +SKIP

        Please see:
        `Table visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
    self.uuid = uuid
    return self