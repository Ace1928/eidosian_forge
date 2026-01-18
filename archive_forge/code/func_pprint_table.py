from __future__ import annotations
import sys
from io import StringIO
from json import JSONEncoder, loads
from typing import TYPE_CHECKING
def pprint_table(table: list[list], out: TextIO=sys.stdout, rstrip: bool=False) -> None:
    """
    Prints out a table of data, padded for alignment
    Each row must have the same number of columns.

    Args:
        table: The table to print. A list of lists.
        out: Output stream (file-like object)
        rstrip: if True, trailing withespaces are removed from the entries.
    """

    def max_width_col(table: list[list], col_idx: int) -> int:
        """Get the maximum width of the given column index."""
        return max((len(row[col_idx]) for row in table))
    if rstrip:
        for row_idx, row in enumerate(table):
            table[row_idx] = [c.rstrip() for c in row]
    col_paddings = []
    ncols = len(table[0])
    for i in range(ncols):
        col_paddings.append(max_width_col(table, i))
    for row in table:
        out.write(row[0].ljust(col_paddings[0] + 1))
        for i in range(1, len(row)):
            col = row[i].rjust(col_paddings[i] + 2)
            out.write(col)
        out.write('\n')