from typing import Tuple
import numpy as np
from modin.core.execution.ray.implementations.cudf_on_ray.partitioning.partition_manager import (
from modin.core.io import CSVDispatcher
@classmethod
def build_partition(cls, partition_ids, row_lengths, column_widths):
    """
        Build array with partitions of `cls.frame_partition_cls` class.

        Parameters
        ----------
        partition_ids : list
            Array with references to the partitions data.
        row_lengths : list
            Partitions rows lengths.
        column_widths : list
            Number of columns in each partition.

        Returns
        -------
        np.ndarray
            Array with shape equals to the shape of `partition_ids` and
            filed with partitions objects.
        """

    def create_partition(i, j):
        return cls.frame_partition_cls(GPU_MANAGERS[i], partition_ids[i][j], length=row_lengths[i], width=column_widths[j])
    return np.array([[create_partition(i, j) for j in range(len(partition_ids[i]))] for i in range(len(partition_ids))])