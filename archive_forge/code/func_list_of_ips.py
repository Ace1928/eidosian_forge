import pandas
from distributed import Future
from distributed.utils import get_ip
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.core.execution.dask.common import DaskWrapper
from modin.utils import _inherit_docstrings
from .partition import PandasOnDaskDataframePartition
@property
def list_of_ips(self):
    """
        Get the IPs holding the physical objects composing this partition.

        Returns
        -------
        List
            A list of IPs as ``distributed.Future`` or str.
        """
    result = [None] * len(self.list_of_block_partitions)
    for idx, partition in enumerate(self.list_of_block_partitions):
        partition.drain_call_queue()
        result[idx] = partition.ip(materialize=False)
    return result