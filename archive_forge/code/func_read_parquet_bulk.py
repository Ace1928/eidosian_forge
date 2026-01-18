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
def read_parquet_bulk(paths: Union[str, List[str]], *, filesystem: Optional['pyarrow.fs.FileSystem']=None, columns: Optional[List[str]]=None, parallelism: int=-1, ray_remote_args: Dict[str, Any]=None, arrow_open_file_args: Optional[Dict[str, Any]]=None, tensor_column_schema: Optional[Dict[str, Tuple[np.dtype, Tuple[int, ...]]]]=None, meta_provider: Optional[BaseFileMetadataProvider]=None, partition_filter: Optional[PathPartitionFilter]=None, shuffle: Union[Literal['files'], None]=None, include_paths: bool=False, file_extensions: Optional[List[str]]=ParquetBaseDatasource._FILE_EXTENSIONS, **arrow_parquet_args) -> Dataset:
    """Create :class:`~ray.data.Dataset` from parquet files without reading metadata.

    Use :meth:`~ray.data.read_parquet` for most cases.

    Use :meth:`~ray.data.read_parquet_bulk` if all the provided paths point to files
    and metadata fetching using :meth:`~ray.data.read_parquet` takes too long or the
    parquet files do not all have a unified schema.

    Performance slowdowns are possible when using this method with parquet files that
    are very large.

    .. warning::

        Only provide file paths as input (i.e., no directory paths). An
        OSError is raised if one or more paths point to directories. If your
        use-case requires directory paths, use :meth:`~ray.data.read_parquet`
        instead.

    Examples:
        Read multiple local files. You should always provide only input file paths
        (i.e. no directory paths) when known to minimize read latency.

        >>> ray.data.read_parquet_bulk( # doctest: +SKIP
        ...     ["/path/to/file1", "/path/to/file2"])

    Args:
        paths: A single file path or a list of file paths.
        filesystem: The PyArrow filesystem
            implementation to read from. These filesystems are
            specified in the
            `PyArrow docs <https://arrow.apache.org/docs/python/api/                filesystems.html#filesystem-implementations>`_.
            Specify this parameter if you need to provide specific configurations to
            the filesystem. By default, the filesystem is automatically selected based
            on the scheme of the paths. For example, if the path begins with ``s3://``,
            the `S3FileSystem` is used.
        columns: A list of column names to read. Only the
            specified columns are read during the file scan.
        parallelism: The amount of parallelism to use for
            the dataset. Defaults to -1, which automatically determines the optimal
            parallelism for your configuration. You should not need to manually set
            this value in most cases. For details on how the parallelism is
            automatically determined and guidance on how to tune it, see
            :ref:`Tuning read parallelism <read_parallelism>`. Parallelism is
            upper bounded by the total number of records in all the parquet files.
        ray_remote_args: kwargs passed to :meth:`~ray.remote` in the read tasks.
        arrow_open_file_args: kwargs passed to
            `pyarrow.fs.FileSystem.open_input_file <https://arrow.apache.org/docs/                python/generated/pyarrow.fs.FileSystem.html                    #pyarrow.fs.FileSystem.open_input_file>`_.
            when opening input files to read.
        tensor_column_schema: A dict of column name to PyArrow dtype and shape
            mappings for converting a Parquet column containing serialized
            tensors (ndarrays) as their elements to PyArrow tensors. This function
            assumes that the tensors are serialized in the raw
            NumPy array format in C-contiguous order (e.g. via
            `arr.tobytes()`).
        meta_provider: A :ref:`file metadata provider <metadata_provider>`. Custom
            metadata providers may be able to resolve file metadata more quickly and/or
            accurately. In most cases, you do not need to set this. If ``None``, this
            function uses a system-chosen implementation.
        partition_filter: A
            :class:`~ray.data.datasource.partitioning.PathPartitionFilter`. Use
            with a custom callback to read only selected partitions of a dataset.
            By default, this filters out any file paths whose file extension does not
            match "*.parquet*".
        shuffle: If setting to "files", randomly shuffle input files order before read.
            Defaults to not shuffle with ``None``.
        arrow_parquet_args: Other parquet read options to pass to PyArrow. For the full
            set of arguments, see
            the `PyArrow API <https://arrow.apache.org/docs/python/generated/                pyarrow.dataset.Scanner.html#pyarrow.dataset.Scanner.from_fragment>`_
        include_paths: If ``True``, include the path to each file. File paths are
            stored in the ``'path'`` column.
        file_extensions: A list of file extensions to filter files by.

    Returns:
       :class:`~ray.data.Dataset` producing records read from the specified paths.
    """
    if meta_provider is None:
        meta_provider = get_parquet_bulk_metadata_provider()
    read_table_args = _resolve_parquet_args(tensor_column_schema, **arrow_parquet_args)
    if columns is not None:
        read_table_args['columns'] = columns
    datasource = ParquetBaseDatasource(paths, read_table_args=read_table_args, filesystem=filesystem, open_stream_args=arrow_open_file_args, meta_provider=meta_provider, partition_filter=partition_filter, shuffle=shuffle, include_paths=include_paths, file_extensions=file_extensions)
    return read_datasource(datasource, parallelism=parallelism, ray_remote_args=ray_remote_args)