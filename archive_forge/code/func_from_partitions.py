from typing import TYPE_CHECKING, Optional, Union
import numpy as np
from pandas._typing import Axes
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.pandas.dataframe import DataFrame, Series
def from_partitions(partitions: list, axis: Optional[int], index: Optional[Axes]=None, columns: Optional[Axes]=None, row_lengths: Optional[list]=None, column_widths: Optional[list]=None) -> DataFrame:
    """
    Create DataFrame from remote partitions.

    Parameters
    ----------
    partitions : list
        A list of Ray.ObjectRef/Dask.Future to partitions depending on the engine used.
        Or a list of tuples of Ray.ObjectRef/Dask.Future to node ip addresses and partitions
        depending on the engine used (i.e. ``[(Ray.ObjectRef/Dask.Future, Ray.ObjectRef/Dask.Future), ...]``).
    axis : {None, 0 or 1}
        The ``axis`` parameter is used to identify what are the partitions passed.
        You have to set:

        * ``axis=0`` if you want to create DataFrame from row partitions
        * ``axis=1`` if you want to create DataFrame from column partitions
        * ``axis=None`` if you want to create DataFrame from 2D list of partitions
    index : sequence, optional
        The index for the DataFrame. Is computed if not provided.
    columns : sequence, optional
        The columns for the DataFrame. Is computed if not provided.
    row_lengths : list, optional
        The length of each partition in the rows. The "height" of
        each of the block partitions. Is computed if not provided.
    column_widths : list, optional
        The width of each partition in the columns. The "width" of
        each of the block partitions. Is computed if not provided.

    Returns
    -------
    modin.pandas.DataFrame
        DataFrame instance created from remote partitions.

    Notes
    -----
    Pass `index`, `columns`, `row_lengths` and `column_widths` to avoid triggering
    extra computations of the metadata when creating a DataFrame.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    factory = FactoryDispatcher.get_factory()
    assert factory is not None
    assert factory.io_cls is not None
    assert factory.io_cls.frame_cls is not None
    assert factory.io_cls.frame_cls._partition_mgr_cls is not None
    partition_class = factory.io_cls.frame_cls._partition_mgr_cls._partition_class
    partition_frame_class = factory.io_cls.frame_cls
    partition_mgr_class = factory.io_cls.frame_cls._partition_mgr_cls
    if axis is None:
        if isinstance(partitions[0][0], tuple):
            parts = np.array([[partition_class(partition, ip=ip) for ip, partition in row] for row in partitions])
        else:
            parts = np.array([[partition_class(partition) for partition in row] for row in partitions])
    elif axis == 0:
        if isinstance(partitions[0], tuple):
            parts = np.array([[partition_class(partition, ip=ip)] for ip, partition in partitions])
        else:
            parts = np.array([[partition_class(partition)] for partition in partitions])
    elif axis == 1:
        if isinstance(partitions[0], tuple):
            parts = np.array([[partition_class(partition, ip=ip) for ip, partition in partitions]])
        else:
            parts = np.array([[partition_class(partition) for partition in partitions]])
    else:
        raise ValueError(f'Got unacceptable value of axis {axis}. Possible values are {0}, {1} or {None}.')
    labels_axis_to_sync = None
    if index is None:
        labels_axis_to_sync = 1
        index, internal_indices = partition_mgr_class.get_indices(0, parts)
        if row_lengths is None:
            row_lengths = [len(idx) for idx in internal_indices]
    if columns is None:
        labels_axis_to_sync = 0 if labels_axis_to_sync is None else -1
        columns, internal_indices = partition_mgr_class.get_indices(1, parts)
        if column_widths is None:
            column_widths = [len(idx) for idx in internal_indices]
    frame = partition_frame_class(parts, index, columns, row_lengths=row_lengths, column_widths=column_widths)
    if labels_axis_to_sync != -1:
        frame.synchronize_labels(axis=labels_axis_to_sync)
    return DataFrame(query_compiler=PandasQueryCompiler(frame))