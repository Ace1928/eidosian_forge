import numpy as np
import pandas
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.file_dispatcher import FileDispatcher
from modin.core.storage_formats.pandas.utils import compute_chunksize
@classmethod
def call_deploy(cls, fname, col_partitions, **kwargs):
    """
        Deploy remote tasks to the workers with passed parameters.

        Parameters
        ----------
        fname : str, path object or file-like object
            Name of the file to read.
        col_partitions : list
            List of arrays with columns names that should be read
            by each partition.
        **kwargs : dict
            Parameters of deploying read_* function.

        Returns
        -------
        np.ndarray
            Array with references to the task deploy result for each partition.
        """
    return np.array([cls.deploy(func=cls.parse, f_kwargs={'fname': fname, 'columns': cols, 'num_splits': NPartitions.get(), **kwargs}, num_returns=NPartitions.get() + 2) for cols in col_partitions]).T