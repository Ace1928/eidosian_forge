import pyarrow as pa
from pyarrow.util import _is_iterable, _stringify_path, _is_path_like
from pyarrow.compute import Expression, scalar, field  # noqa
def parquet_dataset(metadata_path, schema=None, filesystem=None, format=None, partitioning=None, partition_base_dir=None):
    """
    Create a FileSystemDataset from a `_metadata` file created via
    `pyarrow.parquet.write_metadata`.

    Parameters
    ----------
    metadata_path : path,
        Path pointing to a single file parquet metadata file
    schema : Schema, optional
        Optionally provide the Schema for the Dataset, in which case it will
        not be inferred from the source.
    filesystem : FileSystem or URI string, default None
        If a single path is given as source and filesystem is None, then the
        filesystem will be inferred from the path.
        If an URI string is passed, then a filesystem object is constructed
        using the URI's optional path component as a directory prefix. See the
        examples below.
        Note that the URIs on Windows must follow 'file:///C:...' or
        'file:/C:...' patterns.
    format : ParquetFileFormat
        An instance of a ParquetFileFormat if special options needs to be
        passed.
    partitioning : Partitioning, PartitioningFactory, str, list of str
        The partitioning scheme specified with the ``partitioning()``
        function. A flavor string can be used as shortcut, and with a list of
        field names a DirectoryPartitioning will be inferred.
    partition_base_dir : str, optional
        For the purposes of applying the partitioning, paths will be
        stripped of the partition_base_dir. Files not matching the
        partition_base_dir prefix will be skipped for partitioning discovery.
        The ignored files will still be part of the Dataset, but will not
        have partition information.

    Returns
    -------
    FileSystemDataset
        The dataset corresponding to the given metadata
    """
    from pyarrow.fs import LocalFileSystem, _ensure_filesystem
    if format is None:
        format = ParquetFileFormat()
    elif not isinstance(format, ParquetFileFormat):
        raise ValueError('format argument must be a ParquetFileFormat')
    if filesystem is None:
        filesystem = LocalFileSystem()
    else:
        filesystem = _ensure_filesystem(filesystem)
    metadata_path = filesystem.normalize_path(_stringify_path(metadata_path))
    options = ParquetFactoryOptions(partition_base_dir=partition_base_dir, partitioning=_ensure_partitioning(partitioning))
    factory = ParquetDatasetFactory(metadata_path, filesystem, format, options=options)
    return factory.finish(schema)