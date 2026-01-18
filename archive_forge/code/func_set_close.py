from __future__ import annotations
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, formatting, formatting_html, ops
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexed
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import _raise_if_any_duplicate_dimensions
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import is_chunked_array
def set_close(self, close: Callable[[], None] | None) -> None:
    """Register the function that releases any resources linked to this object.

        This method controls how xarray cleans up resources associated
        with this object when the ``.close()`` method is called. It is mostly
        intended for backend developers and it is rarely needed by regular
        end-users.

        Parameters
        ----------
        close : callable
            The function that when called like ``close()`` releases
            any resources linked to this object.
        """
    self._close = close