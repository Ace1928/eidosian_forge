import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from ray.data._internal.null_aggregate import (
from ray.data._internal.sort import SortKey
from ray.data.block import AggType, Block, BlockAccessor, KeyType, T, U
from ray.util.annotations import PublicAPI
def block_row_ls(block: Block) -> AggType:
    block_acc = BlockAccessor.for_block(block)
    ls = []
    for row in block_acc.iter_rows(public_row_format=False):
        ls.append(row.get(on))
    return ls