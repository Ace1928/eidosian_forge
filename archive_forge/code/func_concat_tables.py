import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def concat_tables(tables: List[Table], axis: int=0) -> Table:
    """
    Concatenate tables.

    Args:
        tables (list of `Table`):
            List of tables to be concatenated.
        axis (`{0, 1}`, defaults to `0`, meaning over rows):
            Axis to concatenate over, where `0` means over rows (vertically) and `1` means over columns
            (horizontally).

            <Added version="1.6.0"/>
    Returns:
        `datasets.table.Table`:
            If the number of input tables is > 1, then the returned table is a `datasets.table.ConcatenationTable`.
            Otherwise if there's only one table, it is returned as is.
    """
    tables = list(tables)
    if len(tables) == 1:
        return tables[0]
    return ConcatenationTable.from_tables(tables, axis=axis)