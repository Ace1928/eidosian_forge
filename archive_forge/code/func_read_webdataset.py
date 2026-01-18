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
@PublicAPI(stability='alpha')
def read_webdataset(paths: Union[str, List[str]], *, filesystem: Optional['pyarrow.fs.FileSystem']=None, parallelism: int=-1, arrow_open_stream_args: Optional[Dict[str, Any]]=None, meta_provider: Optional[BaseFileMetadataProvider]=None, partition_filter: Optional[PathPartitionFilter]=None, decoder: Optional[Union[bool, str, callable, list]]=True, fileselect: Optional[Union[list, callable]]=None, filerename: Optional[Union[list, callable]]=None, suffixes: Optional[Union[list, callable]]=None, verbose_open: bool=False, shuffle: Union[Literal['files'], None]=None, include_paths: bool=False, file_extensions: Optional[List[str]]=None) -> Dataset:
    """Create a :class:`~ray.data.Dataset` from
    `WebDataset <https://webdataset.github.io/webdataset/>`_ files.

    Args:
        paths: A single file/directory path or a list of file/directory paths.
            A list of paths can contain both files and directories.
        filesystem: The filesystem implementation to read from.
        parallelism: The requested parallelism of the read. Parallelism may be
            limited by the number of files in the dataset.
        arrow_open_stream_args: Key-word arguments passed to
            `pyarrow.fs.FileSystem.open_input_stream <https://arrow.apache.org/docs/python/generated/pyarrow.fs.FileSystem.html>`_.
            To read a compressed TFRecord file,
            pass the corresponding compression type (e.g. for ``GZIP`` or ``ZLIB``, use
            ``arrow_open_stream_args={'compression_type': 'gzip'}``).
        meta_provider: File metadata provider. Custom metadata providers may
            be able to resolve file metadata more quickly and/or accurately. If
            ``None``, this function uses a system-chosen implementation.
        partition_filter: Path-based partition filter, if any. Can be used
            with a custom callback to read only selected partitions of a dataset.
        decoder: A function or list of functions to decode the data.
        fileselect: A callable or list of glob patterns to select files.
        filerename: A function or list of tuples to rename files prior to grouping.
        suffixes: A function or list of suffixes to select for creating samples.
        verbose_open: Whether to print the file names as they are opened.
        shuffle: If setting to "files", randomly shuffle input files order before read.
            Defaults to not shuffle with ``None``.
        include_paths: If ``True``, include the path to each file. File paths are
            stored in the ``'path'`` column.
        file_extensions: A list of file extensions to filter files by.

    Returns:
        A :class:`~ray.data.Dataset` that contains the example features.

    Raises:
        ValueError: If a file contains a message that isn't a `tf.train.Example`_.

    .. _tf.train.Example: https://www.tensorflow.org/api_docs/python/tf/train/Example
    """
    if meta_provider is None:
        meta_provider = get_generic_metadata_provider(WebDatasetDatasource._FILE_EXTENSIONS)
    datasource = WebDatasetDatasource(paths, decoder=decoder, fileselect=fileselect, filerename=filerename, suffixes=suffixes, verbose_open=verbose_open, filesystem=filesystem, open_stream_args=arrow_open_stream_args, meta_provider=meta_provider, partition_filter=partition_filter, shuffle=shuffle, include_paths=include_paths, file_extensions=file_extensions)
    return read_datasource(datasource, parallelism=parallelism)