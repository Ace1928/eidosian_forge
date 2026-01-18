import cudf
import cupy
import cupy as cp
import numpy as np
import ray
from pandas.core.dtypes.common import is_list_like
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.ray.common import RayWrapper
from modin.core.execution.ray.common.utils import ObjectIDType
def apply_result_not_dataframe(self, func, **kwargs):
    """
        Apply `func` to this partition.

        Parameters
        ----------
        func : callable
            A function to apply.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        ray.ObjectRef
            A reference to integer key of result
            in internal dict-storage of `self.gpu_manager`.
        """
    return self.gpu_manager.apply_result_not_dataframe.remote(self.get_key(), func, **kwargs)