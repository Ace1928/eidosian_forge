import cudf
import cupy
import cupy as cp
import numpy as np
import ray
from pandas.core.dtypes.common import is_list_like
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.ray.common import RayWrapper
from modin.core.execution.ray.common.utils import ObjectIDType
def get_object_id(self):
    """
        Get object stored for this partition from `self.gpu_manager`.

        Returns
        -------
        ray.ObjectRef
        """
    return self.gpu_manager.get_object_id.remote(self.get_key())