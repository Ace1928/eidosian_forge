import io
import pathlib
import posixpath
import warnings
from typing import (
import numpy as np
import ray
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.util import (
from ray.data.block import Block, BlockAccessor
from ray.data.context import DataContext
from ray.data.datasource.block_path_provider import BlockWritePathProvider
from ray.data.datasource.datasource import Datasource, ReadTask, WriteResult
from ray.data.datasource.file_meta_provider import (
from ray.data.datasource.filename_provider import (
from ray.data.datasource.partitioning import (
from ray.data.datasource.path_util import (
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
def create_read_task_fn(read_paths, num_threads):

    def read_task_fn():
        nonlocal num_threads, read_paths
        if num_threads > 0:
            if len(read_paths) < num_threads:
                num_threads = len(read_paths)
            logger.get_logger().debug(f'Reading {len(read_paths)} files with {num_threads} threads.')
            yield from make_async_gen(iter(read_paths), read_files, num_workers=num_threads)
        else:
            logger.get_logger().debug(f'Reading {len(read_paths)} files.')
            yield from read_files(read_paths)
    return read_task_fn