import collections
from collections.abc import Iterable
import functools
import math
import warnings
from affine import Affine
import attr
import numpy as np
from rasterio.errors import WindowError, RasterioDeprecationWarning
from rasterio.transform import rowcol, guard_transform
@classmethod
def from_slices(cls, rows, cols, height=-1, width=-1, boundless=False):
    """Construct a Window from row and column slices or tuples / lists of
        start and stop indexes. Converts the rows and cols to offsets, height,
        and width.

        In general, indexes are defined relative to the upper left corner of
        the dataset: rows=(0, 10), cols=(0, 4) defines a window that is 4
        columns wide and 10 rows high starting from the upper left.

        Start indexes may be `None` and will default to 0.
        Stop indexes may be `None` and will default to width or height, which
        must be provided in this case.

        Negative start indexes are evaluated relative to the lower right of the
        dataset: rows=(-2, None), cols=(-2, None) defines a window that is 2
        rows high and 2 columns wide starting from the bottom right.

        Parameters
        ----------
        rows, cols: slice, tuple, or list
            Slices or 2 element tuples/lists containing start, stop indexes.
        height, width: float
            A shape to resolve relative values against. Only used when a start
            or stop index is negative or a stop index is None.
        boundless: bool, optional
            Whether the inputs are bounded (default) or not.

        Returns
        -------
        Window
        """
    if isinstance(rows, (tuple, list)):
        if len(rows) != 2:
            raise WindowError('rows must have a start and stop index')
        rows = slice(*rows)
    elif not isinstance(rows, slice):
        raise WindowError('rows must be a slice, tuple, or list')
    if isinstance(cols, (tuple, list)):
        if len(cols) != 2:
            raise WindowError('cols must have a start and stop index')
        cols = slice(*cols)
    elif not isinstance(cols, slice):
        raise WindowError('cols must be a slice, tuple, or list')
    if rows.stop is None and height < 0:
        raise WindowError('height is required if row stop index is None')
    if cols.stop is None and width < 0:
        raise WindowError('width is required if col stop index is None')
    row_off = 0.0 if rows.start is None else rows.start
    row_stop = height if rows.stop is None else rows.stop
    col_off = 0.0 if cols.start is None else cols.start
    col_stop = width if cols.stop is None else cols.stop
    if not boundless:
        if row_off < 0 or row_stop < 0:
            if height < 0:
                raise WindowError('height is required when providing negative indexes')
            if row_off < 0:
                row_off += height
            if row_stop < 0:
                row_stop += height
        if col_off < 0 or col_stop < 0:
            if width < 0:
                raise WindowError('width is required when providing negative indexes')
            if col_off < 0:
                col_off += width
            if col_stop < 0:
                col_stop += width
    num_cols = max(col_stop - col_off, 0.0)
    num_rows = max(row_stop - row_off, 0.0)
    return cls(col_off=col_off, row_off=row_off, width=num_cols, height=num_rows)