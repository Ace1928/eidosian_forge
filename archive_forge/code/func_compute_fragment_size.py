from typing import List, Optional, Tuple, Union
import pyarrow as pa
import pyhdk
from packaging import version
from pyhdk.hdk import HDK, ExecutionResult, QueryNode, RelAlgExecutor
from modin.config import CpuCount, HdkFragmentSize, HdkLaunchParameters
from modin.utils import _inherit_docstrings
from .base_worker import BaseDbWorker, DbTable
@classmethod
def compute_fragment_size(cls, table):
    """
        Compute fragment size to be used for table import.

        Parameters
        ----------
        table : pyarrow.Table
            A table to import.

        Returns
        -------
        int
            Fragment size to use for import.
        """
    fragment_size = HdkFragmentSize.get()
    if fragment_size is None:
        if cls._preferred_device == 'CPU':
            cpu_count = CpuCount.get()
            if cpu_count is not None:
                fragment_size = table.num_rows // cpu_count
                fragment_size = min(fragment_size, 2 ** 25)
                fragment_size = max(fragment_size, 2 ** 18)
            else:
                fragment_size = 0
        else:
            fragment_size = 2 ** 25
    else:
        fragment_size = int(fragment_size)
    return fragment_size