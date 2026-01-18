import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def pandas_df_to_arrow_block(df: 'pandas.DataFrame') -> 'Block':
    from ray.data.block import BlockAccessor, BlockExecStats
    stats = BlockExecStats.builder()
    import pyarrow as pa
    block = pa.table(df)
    return (block, BlockAccessor.for_block(block).get_metadata(input_files=None, exec_stats=stats.build()))