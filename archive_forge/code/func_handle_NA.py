import numpy as np
from patsy import PatsyError
from patsy.util import (safe_isnan, safe_scalar_isnan,
def handle_NA(self, values, is_NAs, origins):
    """Takes a set of factor values that may have NAs, and handles them
        appropriately.

        :arg values: A list of `ndarray` objects representing the data.
          These may be 1- or 2-dimensional, and may be of varying dtype. All
          will have the same number of rows (or entries, for 1-d arrays).
        :arg is_NAs: A list with the same number of entries as `values`,
          containing boolean `ndarray` objects that indicate which rows
          contain NAs in the corresponding entry in `values`.
        :arg origins: A list with the same number of entries as
          `values`, containing information on the origin of each
          value. If we encounter a problem with some particular value, we use
          the corresponding entry in `origins` as the origin argument when
          raising a :class:`PatsyError`.
        :returns: A list of new values (which may have a differing number of
          rows.)
        """
    assert len(values) == len(is_NAs) == len(origins)
    if len(values) == 0:
        return values
    if self.on_NA == 'raise':
        return self._handle_NA_raise(values, is_NAs, origins)
    elif self.on_NA == 'drop':
        return self._handle_NA_drop(values, is_NAs, origins)
    else:
        assert False