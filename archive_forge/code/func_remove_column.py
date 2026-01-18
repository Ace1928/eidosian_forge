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
def remove_column(self, i, *args, **kwargs):
    """
        Create new Table with the indicated column removed.

        Args:
            i (`int`):
                Index of column to remove.

        Returns:
            `datasets.table.Table`:
                New table without the column.
        """
    table = self.table.remove_column(i, *args, **kwargs)
    name = self.table.column_names[i]
    blocks = []
    for tables in self.blocks:
        blocks.append([t.remove_column(t.column_names.index(name), *args, **kwargs) if name in t.column_names else t for t in tables])
    return ConcatenationTable(table, blocks)