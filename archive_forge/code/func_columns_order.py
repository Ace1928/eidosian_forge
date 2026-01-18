from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
@property
def columns_order(self) -> Optional[dict[int, IndexLabel]]:
    """
        Get order of columns for the described dataframe if available.

        Returns
        -------
        dict[int, IndexLabel] or None
        """
    if self._columns_order is not None:
        return self._columns_order
    if self._parent_df is None or not self._parent_df.has_materialized_columns:
        return None
    actual_columns = self._parent_df.columns
    self._normalize_self_levels(actual_columns)
    self._columns_order = {i: col for i, col in enumerate(actual_columns)}
    if len(self._columns_order) > len(self._known_dtypes) + len(self._cols_with_unknown_dtypes):
        new_cols = [col for col in self._columns_order.values() if col not in self._known_dtypes and col not in self._cols_with_unknown_dtypes]
        if self._remaining_dtype is not None:
            self._known_dtypes.update({col: self._remaining_dtype for col in new_cols})
            self._remaining_dtype = None
            if len(self._cols_with_unknown_dtypes) == 0:
                self._schema_is_known = True
        else:
            self._cols_with_unknown_dtypes.extend(new_cols)
    self._know_all_names = True
    return self._columns_order