import os
import warnings
from abc import ABC
from functools import wraps
from typing import TYPE_CHECKING
import numpy as np
import pandas
from pandas._libs.lib import no_default
from modin.config import (
from modin.core.dataframe.pandas.utils import create_pandas_df_from_partitions
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
def get_right_block(right_partitions, row_idx, col_idx):
    partition = right_partitions[row_idx][col_idx]
    blocks = partition.list_of_blocks
    "\n            NOTE:\n            Currently we do one remote call per right virtual partition to\n            materialize the partitions' blocks, then another remote call to do\n            the n_ary operation. we could get better performance if we\n            assembled the other partition within the remote `apply` call, by\n            passing the partition in as `other_axis_partition`. However,\n            passing `other_axis_partition` requires some extra care that would\n            complicate the code quite a bit:\n            - block partitions don't know how to deal with `other_axis_partition`\n            - the right axis partition's axis could be different from the axis\n              of the corresponding left partition\n            - there can be multiple other_axis_partition because this is an n-ary\n              operation and n can be > 2.\n            So for now just do the materialization in a separate remote step.\n            "
    if len(blocks) > 1:
        partition.force_materialization()
    assert len(partition.list_of_blocks) == 1
    return partition.list_of_blocks[0]