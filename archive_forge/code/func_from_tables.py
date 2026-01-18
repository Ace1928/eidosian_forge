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
@classmethod
def from_tables(cls, tables: List[Union[pa.Table, Table]], axis: int=0) -> 'ConcatenationTable':
    """Create `ConcatenationTable` from list of tables.

        Args:
            tables (list of `Table` or list of `pyarrow.Table`):
                List of tables.
            axis (`{0, 1}`, defaults to `0`, meaning over rows):
                Axis to concatenate over, where `0` means over rows (vertically) and `1` means over columns
                (horizontally).

                <Added version="1.6.0"/>
        """

    def to_blocks(table: Union[pa.Table, Table]) -> List[List[TableBlock]]:
        if isinstance(table, pa.Table):
            return [[InMemoryTable(table)]]
        elif isinstance(table, ConcatenationTable):
            return copy.deepcopy(table.blocks)
        else:
            return [[table]]

    def _slice_row_block(row_block: List[TableBlock], length: int) -> Tuple[List[TableBlock], List[TableBlock]]:
        sliced = [table.slice(0, length) for table in row_block]
        remainder = [table.slice(length, len(row_block[0]) - length) for table in row_block]
        return (sliced, remainder)

    def _split_both_like(result: List[List[TableBlock]], blocks: List[List[TableBlock]]) -> Tuple[List[List[TableBlock]], List[List[TableBlock]]]:
        """
            Make sure each row_block contain the same num_rows to be able to concatenate them on axis=1.

            To do so, we modify both blocks sets to have the same row_blocks boundaries.
            For example, if `result` has 2 row_blocks of 3 rows and `blocks` has 3 row_blocks of 2 rows,
            we modify both to have 4 row_blocks of size 2, 1, 1 and 2:

                    [ x   x   x | x   x   x ]
                +   [ y   y | y   y | y   y ]
                -----------------------------
                =   [ x   x | x | x | x   x ]
                    [ y   y | y | y | y   y ]

            """
        result, blocks = (list(result), list(blocks))
        new_result, new_blocks = ([], [])
        while result and blocks:
            if len(result[0][0]) > len(blocks[0][0]):
                new_blocks.append(blocks[0])
                sliced, result[0] = _slice_row_block(result[0], len(blocks.pop(0)[0]))
                new_result.append(sliced)
            elif len(result[0][0]) < len(blocks[0][0]):
                new_result.append(result[0])
                sliced, blocks[0] = _slice_row_block(blocks[0], len(result.pop(0)[0]))
                new_blocks.append(sliced)
            else:
                new_result.append(result.pop(0))
                new_blocks.append(blocks.pop(0))
        if result or blocks:
            raise ValueError("Failed to concatenate on axis=1 because tables don't have the same number of rows")
        return (new_result, new_blocks)

    def _extend_blocks(result: List[List[TableBlock]], blocks: List[List[TableBlock]], axis: int=0) -> List[List[TableBlock]]:
        if axis == 0:
            result.extend(blocks)
        elif axis == 1:
            result, blocks = _split_both_like(result, blocks)
            for i, row_block in enumerate(blocks):
                result[i].extend(row_block)
        return result
    blocks = to_blocks(tables[0])
    for table in tables[1:]:
        table_blocks = to_blocks(table)
        blocks = _extend_blocks(blocks, table_blocks, axis=axis)
    return cls.from_blocks(blocks)