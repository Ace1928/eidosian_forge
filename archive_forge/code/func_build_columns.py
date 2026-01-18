import numpy as np
import pandas
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.file_dispatcher import FileDispatcher
from modin.core.storage_formats.pandas.utils import compute_chunksize
@classmethod
def build_columns(cls, columns, num_row_parts=None):
    """
        Split columns into chunks that should be read by workers.

        Parameters
        ----------
        columns : list
            List of columns that should be read from file.
        num_row_parts : int, optional
            Number of parts the dataset is split into. This parameter is used
            to align the column partitioning with it so we won't end up with an
            over partitioned frame.

        Returns
        -------
        col_partitions : list
            List of lists with columns for reading by workers.
        column_widths : list
            List with lengths of `col_partitions` subarrays
            (number of columns that should be read by workers).
        """
    columns_length = len(columns)
    if columns_length == 0:
        return ([], [])
    if num_row_parts is None:
        min_block_size = 1
    else:
        num_remaining_parts = round(NPartitions.get() / num_row_parts)
        min_block_size = min(columns_length // num_remaining_parts, MinPartitionSize.get())
    column_splits = compute_chunksize(columns_length, NPartitions.get(), max(1, min_block_size))
    col_partitions = [columns[i:i + column_splits] for i in range(0, columns_length, column_splits)]
    column_widths = [len(c) for c in col_partitions]
    return (col_partitions, column_widths)