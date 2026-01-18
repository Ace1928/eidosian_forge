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
def read_tfrecords(paths: Union[str, List[str]], *, filesystem: Optional['pyarrow.fs.FileSystem']=None, parallelism: int=-1, arrow_open_stream_args: Optional[Dict[str, Any]]=None, meta_provider: Optional[BaseFileMetadataProvider]=None, partition_filter: Optional[PathPartitionFilter]=None, include_paths: bool=False, ignore_missing_paths: bool=False, tf_schema: Optional['schema_pb2.Schema']=None, shuffle: Union[Literal['files'], None]=None, file_extensions: Optional[List[str]]=None) -> Dataset:
    """Create a :class:`~ray.data.Dataset` from TFRecord files that contain
    `tf.train.Example <https://www.tensorflow.org/api_docs/python/tf/train/Example>`_
    messages.

    .. warning::
        This function exclusively supports ``tf.train.Example`` messages. If a file
        contains a message that isn't of type ``tf.train.Example``, then this function
        fails.

    Examples:
        >>> import ray
        >>> ray.data.read_tfrecords("s3://anonymous@ray-example-data/iris.tfrecords")
        Dataset(
           num_blocks=...,
           num_rows=150,
           schema={...}
        )

        We can also read compressed TFRecord files, which use one of the
        `compression types supported by Arrow <https://arrow.apache.org/docs/python/            generated/pyarrow.CompressedInputStream.html>`_:

        >>> ray.data.read_tfrecords(
        ...     "s3://anonymous@ray-example-data/iris.tfrecords.gz",
        ...     arrow_open_stream_args={"compression": "gzip"},
        ... )
        Dataset(
           num_blocks=...,
           num_rows=150,
           schema={...}
        )

    Args:
        paths: A single file or directory, or a list of file or directory paths.
            A list of paths can contain both files and directories.
        filesystem: The PyArrow filesystem
            implementation to read from. These filesystems are specified in the
            `PyArrow docs <https://arrow.apache.org/docs/python/api/            filesystems.html#filesystem-implementations>`_. Specify this parameter if
            you need to provide specific configurations to the filesystem. By default,
            the filesystem is automatically selected based on the scheme of the paths.
            For example, if the path begins with ``s3://``, the `S3FileSystem` is used.
        parallelism: The amount of parallelism to use for the dataset. Defaults to -1,
            which automatically determines the optimal parallelism for your
            configuration. You should not need to manually set this value in most cases.
            For details on how the parallelism is automatically determined and guidance
            on how to tune it, see :ref:`Tuning read parallelism
            <read_parallelism>`. Parallelism is upper bounded by the total number of
            records in all the CSV files.
        arrow_open_stream_args: kwargs passed to
            `pyarrow.fs.FileSystem.open_input_file <https://arrow.apache.org/docs/                python/generated/pyarrow.fs.FileSystem.html                    #pyarrow.fs.FileSystem.open_input_stream>`_.
            when opening input files to read. To read a compressed TFRecord file,
            pass the corresponding compression type (e.g., for ``GZIP`` or ``ZLIB``),
            use ``arrow_open_stream_args={'compression_type': 'gzip'}``).
        meta_provider: A :ref:`file metadata provider <metadata_provider>`. Custom
            metadata providers may be able to resolve file metadata more quickly and/or
            accurately. In most cases, you do not need to set this. If ``None``, this
            function uses a system-chosen implementation.
        partition_filter: A
            :class:`~ray.data.datasource.partitioning.PathPartitionFilter`.
            Use with a custom callback to read only selected partitions of a
            dataset.
        include_paths: If ``True``, include the path to each file. File paths are
            stored in the ``'path'`` column.
        ignore_missing_paths:  If True, ignores any file paths in ``paths`` that are not
            found. Defaults to False.
        tf_schema: Optional TensorFlow Schema which is used to explicitly set the schema
            of the underlying Dataset.
        shuffle: If setting to "files", randomly shuffle input files order before read.
            Defaults to not shuffle with ``None``.
        file_extensions: A list of file extensions to filter files by.

    Returns:
        A :class:`~ray.data.Dataset` that contains the example features.

    Raises:
        ValueError: If a file contains a message that isn't a ``tf.train.Example``.
    """
    if meta_provider is None:
        meta_provider = get_generic_metadata_provider(TFRecordDatasource._FILE_EXTENSIONS)
    datasource = TFRecordDatasource(paths, tf_schema=tf_schema, filesystem=filesystem, open_stream_args=arrow_open_stream_args, meta_provider=meta_provider, partition_filter=partition_filter, ignore_missing_paths=ignore_missing_paths, shuffle=shuffle, include_paths=include_paths, file_extensions=file_extensions)
    return read_datasource(datasource, parallelism=parallelism)