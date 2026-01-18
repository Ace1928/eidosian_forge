import collections
import logging
import os
from typing import (
import numpy as np
import ray
from ray._private.auto_init_hook import wrap_auto_init
from ray.air.util.tensor_extensions.utils import _create_possibly_ragged_ndarray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.from_operators import (
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.optimizers import LogicalPlan
from ray.data._internal.plan import ExecutionPlan
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats
from ray.data._internal.util import (
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.data.dataset import Dataset, MaterializedDataset
from ray.data.datasource import (
from ray.data.datasource._default_metadata_providers import (
from ray.data.datasource.datasource import Reader
from ray.data.datasource.file_based_datasource import (
from ray.data.datasource.partitioning import Partitioning
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
@PublicAPI
def read_numpy(paths: Union[str, List[str]], *, filesystem: Optional['pyarrow.fs.FileSystem']=None, parallelism: int=-1, arrow_open_stream_args: Optional[Dict[str, Any]]=None, meta_provider: Optional[BaseFileMetadataProvider]=None, partition_filter: Optional[PathPartitionFilter]=None, partitioning: Partitioning=None, include_paths: bool=False, ignore_missing_paths: bool=False, shuffle: Union[Literal['files'], None]=None, file_extensions: Optional[List[str]]=NumpyDatasource._FILE_EXTENSIONS, **numpy_load_args) -> Dataset:
    """Create an Arrow dataset from numpy files.

    Examples:
        Read a directory of files in remote storage.

        >>> import ray
        >>> ray.data.read_numpy("s3://bucket/path") # doctest: +SKIP

        Read multiple local files.

        >>> ray.data.read_numpy(["/path/to/file1", "/path/to/file2"]) # doctest: +SKIP

        Read multiple directories.

        >>> ray.data.read_numpy( # doctest: +SKIP
        ...     ["s3://bucket/path1", "s3://bucket/path2"])

    Args:
        paths: A single file/directory path or a list of file/directory paths.
            A list of paths can contain both files and directories.
        filesystem: The filesystem implementation to read from.
        parallelism: The requested parallelism of the read. Parallelism may be
            limited by the number of files of the dataset.
        arrow_open_stream_args: kwargs passed to
            `pyarrow.fs.FileSystem.open_input_stream <https://arrow.apache.org/docs/python/generated/pyarrow.fs.FileSystem.html>`_.
        numpy_load_args: Other options to pass to np.load.
        meta_provider: File metadata provider. Custom metadata providers may
            be able to resolve file metadata more quickly and/or accurately. If
            ``None``, this function uses a system-chosen implementation.
        partition_filter: Path-based partition filter, if any. Can be used
            with a custom callback to read only selected partitions of a dataset.
            By default, this filters out any file paths whose file extension does not
            match "*.npy*".
        partitioning: A :class:`~ray.data.datasource.partitioning.Partitioning` object
            that describes how paths are organized. Defaults to ``None``.
        include_paths: If ``True``, include the path to each file. File paths are
            stored in the ``'path'`` column.
        ignore_missing_paths: If True, ignores any file paths in ``paths`` that are not
            found. Defaults to False.
        shuffle: If setting to "files", randomly shuffle input files order before read.
            Defaults to not shuffle with ``None``.
        file_extensions: A list of file extensions to filter files by.

    Returns:
        Dataset holding Tensor records read from the specified paths.
    """
    if meta_provider is None:
        meta_provider = get_generic_metadata_provider(NumpyDatasource._FILE_EXTENSIONS)
    datasource = NumpyDatasource(paths, numpy_load_args=numpy_load_args, filesystem=filesystem, open_stream_args=arrow_open_stream_args, meta_provider=meta_provider, partition_filter=partition_filter, partitioning=partitioning, ignore_missing_paths=ignore_missing_paths, shuffle=shuffle, include_paths=include_paths, file_extensions=file_extensions)
    return read_datasource(datasource, parallelism=parallelism)