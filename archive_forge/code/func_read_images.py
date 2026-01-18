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
@PublicAPI(stability='beta')
def read_images(paths: Union[str, List[str]], *, filesystem: Optional['pyarrow.fs.FileSystem']=None, parallelism: int=-1, meta_provider: Optional[BaseFileMetadataProvider]=None, ray_remote_args: Dict[str, Any]=None, arrow_open_file_args: Optional[Dict[str, Any]]=None, partition_filter: Optional[PathPartitionFilter]=None, partitioning: Partitioning=None, size: Optional[Tuple[int, int]]=None, mode: Optional[str]=None, include_paths: bool=False, ignore_missing_paths: bool=False, shuffle: Union[Literal['files'], None]=None, file_extensions: Optional[List[str]]=ImageDatasource._FILE_EXTENSIONS) -> Dataset:
    """Creates a :class:`~ray.data.Dataset` from image files.

    Examples:
        >>> import ray
        >>> path = "s3://anonymous@ray-example-data/batoidea/JPEGImages/"
        >>> ds = ray.data.read_images(path)
        >>> ds.schema()
        Column  Type
        ------  ----
        image   numpy.ndarray(shape=(32, 32, 3), dtype=uint8)

        If you need image file paths, set ``include_paths=True``.

        >>> ds = ray.data.read_images(path, include_paths=True)
        >>> ds.schema()
        Column  Type
        ------  ----
        image   numpy.ndarray(shape=(32, 32, 3), dtype=uint8)
        path    string
        >>> ds.take(1)[0]["path"]
        'ray-example-data/batoidea/JPEGImages/1.jpeg'

        If your images are arranged like:

        .. code::

            root/dog/xxx.png
            root/dog/xxy.png

            root/cat/123.png
            root/cat/nsdf3.png

        Then you can include the labels by specifying a
        :class:`~ray.data.datasource.partitioning.Partitioning`.

        >>> import ray
        >>> from ray.data.datasource.partitioning import Partitioning
        >>> root = "s3://anonymous@ray-example-data/image-datasets/dir-partitioned"
        >>> partitioning = Partitioning("dir", field_names=["class"], base_dir=root)
        >>> ds = ray.data.read_images(root, size=(224, 224), partitioning=partitioning)
        >>> ds.schema()
        Column  Type
        ------  ----
        image   numpy.ndarray(shape=(224, 224, 3), dtype=uint8)
        class   string

    Args:
        paths: A single file or directory, or a list of file or directory paths.
            A list of paths can contain both files and directories.
        filesystem: The pyarrow filesystem
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
        meta_provider: A :ref:`file metadata provider <metadata_provider>`. Custom
            metadata providers may be able to resolve file metadata more quickly and/or
            accurately. In most cases, you do not need to set this. If ``None``, this
            function uses a system-chosen implementation.
        ray_remote_args: kwargs passed to :meth:`~ray.remote` in the read tasks.
        arrow_open_file_args: kwargs passed to
            `pyarrow.fs.FileSystem.open_input_file <https://arrow.apache.org/docs/                python/generated/pyarrow.fs.FileSystem.html                    #pyarrow.fs.FileSystem.open_input_file>`_.
            when opening input files to read.
        partition_filter:  A
            :class:`~ray.data.datasource.partitioning.PathPartitionFilter`. Use
            with a custom callback to read only selected partitions of a dataset.
            By default, this filters out any file paths whose file extension does not
            match ``*.png``, ``*.jpg``, ``*.jpeg``, ``*.tiff``, ``*.bmp``, or ``*.gif``.
        partitioning: A :class:`~ray.data.datasource.partitioning.Partitioning` object
            that describes how paths are organized. Defaults to ``None``.
        size: The desired height and width of loaded images. If unspecified, images
            retain their original shape.
        mode: A `Pillow mode <https://pillow.readthedocs.io/en/stable/handbook/concepts            .html#modes>`_
            describing the desired type and depth of pixels. If unspecified, image
            modes are inferred by
            `Pillow <https://pillow.readthedocs.io/en/stable/index.html>`_.
        include_paths: If ``True``, include the path to each image. File paths are
            stored in the ``'path'`` column.
        ignore_missing_paths: If True, ignores any file/directory paths in ``paths``
            that are not found. Defaults to False.
        shuffle: If setting to "files", randomly shuffle input files order before read.
            Defaults to not shuffle with ``None``.
        file_extensions: A list of file extensions to filter files by.

    Returns:
        A :class:`~ray.data.Dataset` producing tensors that represent the images at
        the specified paths. For information on working with tensors, read the
        :ref:`tensor data guide <working_with_tensors>`.

    Raises:
        ValueError: if ``size`` contains non-positive numbers.
        ValueError: if ``mode`` is unsupported.
    """
    if meta_provider is None:
        meta_provider = get_image_metadata_provider()
    datasource = ImageDatasource(paths, size=size, mode=mode, include_paths=include_paths, filesystem=filesystem, meta_provider=meta_provider, open_stream_args=arrow_open_file_args, partition_filter=partition_filter, partitioning=partitioning, ignore_missing_paths=ignore_missing_paths, shuffle=shuffle, file_extensions=file_extensions)
    return read_datasource(datasource, parallelism=parallelism, ray_remote_args=ray_remote_args)