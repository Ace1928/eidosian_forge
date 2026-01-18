from __future__ import annotations
from collections.abc import (
from typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
def fill_mi_header(row: list[Hashable], control_row: list[bool]) -> tuple[list[Hashable], list[bool]]:
    """
    Forward fill blank entries in row but only inside the same parent index.

    Used for creating headers in Multiindex.

    Parameters
    ----------
    row : list
        List of items in a single row.
    control_row : list of bool
        Helps to determine if particular column is in same parent index as the
        previous value. Used to stop propagation of empty cells between
        different indexes.

    Returns
    -------
    Returns changed row and control_row
    """
    last = row[0]
    for i in range(1, len(row)):
        if not control_row[i]:
            last = row[i]
        if row[i] == '' or row[i] is None:
            row[i] = last
        else:
            control_row[i] = False
            last = row[i]
    return (row, control_row)