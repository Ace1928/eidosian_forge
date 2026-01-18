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
def read_csv(paths: Union[str, List[str]], *, filesystem: Optional['pyarrow.fs.FileSystem']=None, parallelism: int=-1, ray_remote_args: Dict[str, Any]=None, arrow_open_stream_args: Optional[Dict[str, Any]]=None, meta_provider: Optional[BaseFileMetadataProvider]=None, partition_filter: Optional[PathPartitionFilter]=None, partitioning: Partitioning=Partitioning('hive'), include_paths: bool=False, ignore_missing_paths: bool=False, shuffle: Union[Literal['files'], None]=None, file_extensions: Optional[List[str]]=None, **arrow_csv_args) -> Dataset:
    """Creates a :class:`~ray.data.Dataset` from CSV files.

    Examples:
        Read a file in remote storage.

        >>> import ray
        >>> ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")
        >>> ds.schema()
        Column             Type
        ------             ----
        sepal length (cm)  double
        sepal width (cm)   double
        petal length (cm)  double
        petal width (cm)   double
        target             int64

        Read multiple local files.

        >>> ray.data.read_csv( # doctest: +SKIP
        ...    ["local:///path/to/file1", "local:///path/to/file2"])

        Read a directory from remote storage.

        >>> ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris-csv/")

        Read files that use a different delimiter. For more uses of ParseOptions see
        https://arrow.apache.org/docs/python/generated/pyarrow.csv.ParseOptions.html  # noqa: #501

        >>> from pyarrow import csv
        >>> parse_options = csv.ParseOptions(delimiter="\\t")
        >>> ds = ray.data.read_csv(
        ...     "s3://anonymous@ray-example-data/iris.tsv",
        ...     parse_options=parse_options)
        >>> ds.schema()
        Column        Type
        ------        ----
        sepal.length  double
        sepal.width   double
        petal.length  double
        petal.width   double
        variety       string

        Convert a date column with a custom format from a CSV file. For more uses of ConvertOptions see https://arrow.apache.org/docs/python/generated/pyarrow.csv.ConvertOptions.html  # noqa: #501

        >>> from pyarrow import csv
        >>> convert_options = csv.ConvertOptions(
        ...     timestamp_parsers=["%m/%d/%Y"])
        >>> ds = ray.data.read_csv(
        ...     "s3://anonymous@ray-example-data/dow_jones.csv",
        ...     convert_options=convert_options)

        By default, :meth:`~ray.data.read_csv` parses
        `Hive-style partitions <https://athena.guide/        articles/hive-style-partitioning/>`_
        from file paths. If your data adheres to a different partitioning scheme, set
        the ``partitioning`` parameter.

        >>> ds = ray.data.read_csv("s3://anonymous@ray-example-data/year=2022/month=09/sales.csv")
        >>> ds.take(1)
        [{'order_number': 10107, 'quantity': 30, 'year': '2022', 'month': '09'}]

        By default, :meth:`~ray.data.read_csv` reads all files from file paths. If you want to filter
        files by file extensions, set the ``partition_filter`` parameter.

        Read only ``*.csv`` files from a directory.

        >>> ray.data.read_csv("s3://anonymous@ray-example-data/different-extensions/",
        ...     file_extensions=["csv"])
        Dataset(num_blocks=..., num_rows=1, schema={a: int64, b: int64})

    Args:
        paths: A single file or directory, or a list of file or directory paths.
            A list of paths can contain both files and directories.
        filesystem: The PyArrow filesystem
            implementation to read from. These filesystems are specified in the
            `pyarrow docs <https://arrow.apache.org/docs/python/api/            filesystems.html#filesystem-implementations>`_. Specify this parameter if
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
        ray_remote_args: kwargs passed to :meth:`~ray.remote` in the read tasks.
        arrow_open_stream_args: kwargs passed to
            `pyarrow.fs.FileSystem.open_input_file <https://arrow.apache.org/docs/                python/generated/pyarrow.fs.FileSystem.html                    #pyarrow.fs.FileSystem.open_input_stream>`_.
            when opening input files to read.
        meta_provider: A :ref:`file metadata provider <metadata_provider>`. Custom
            metadata providers may be able to resolve file metadata more quickly and/or
            accurately. In most cases, you do not need to set this. If ``None``, this
            function uses a system-chosen implementation.
        partition_filter: A
            :class:`~ray.data.datasource.partitioning.PathPartitionFilter`.
            Use with a custom callback to read only selected partitions of a
            dataset. By default, no files are filtered.
        partitioning: A :class:`~ray.data.datasource.partitioning.Partitioning` object
            that describes how paths are organized. By default, this function parses
            `Hive-style partitions <https://athena.guide/articles/                hive-style-partitioning/>`_.
        include_paths: If ``True``, include the path to each file. File paths are
            stored in the ``'path'`` column.
        ignore_missing_paths: If True, ignores any file paths in ``paths`` that are not
            found. Defaults to False.
        shuffle: If setting to "files", randomly shuffle input files order before read.
            Defaults to not shuffle with ``None``.
        arrow_csv_args: CSV read options to pass to
            `pyarrow.csv.open_csv <https://arrow.apache.org/docs/python/generated/            pyarrow.csv.open_csv.html#pyarrow.csv.open_csv>`_
            when opening CSV files.
        file_extensions: A list of file extensions to filter files by.

    Returns:
        :class:`~ray.data.Dataset` producing records read from the specified paths.
    """
    if meta_provider is None:
        meta_provider = get_generic_metadata_provider(CSVDatasource._FILE_EXTENSIONS)
    datasource = CSVDatasource(paths, arrow_csv_args=arrow_csv_args, filesystem=filesystem, open_stream_args=arrow_open_stream_args, meta_provider=meta_provider, partition_filter=partition_filter, partitioning=partitioning, ignore_missing_paths=ignore_missing_paths, shuffle=shuffle, include_paths=include_paths, file_extensions=file_extensions)
    return read_datasource(datasource, parallelism=parallelism, ray_remote_args=ray_remote_args)