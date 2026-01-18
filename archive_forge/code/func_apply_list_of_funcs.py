import pandas
from distributed import Future
from distributed.utils import get_ip
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.dask.common import DaskWrapper
from modin.logging import get_logger
from modin.pandas.indexing import compute_sliced_len
def apply_list_of_funcs(call_queue, partition):
    """
    Execute all operations stored in the call queue on the partition in a worker process.

    Parameters
    ----------
    call_queue : list
        A call queue of ``[func, args, kwargs]`` triples that needs to be executed on the partition.
    partition : pandas.DataFrame
        A pandas DataFrame the call queue needs to be executed on.

    Returns
    -------
    pandas.DataFrame
        The resulting pandas DataFrame.
    str
        The node IP address of the worker process.
    """
    for func, f_args, f_kwargs in call_queue:
        partition = func(partition, *f_args, **f_kwargs)
    return (partition, get_ip())