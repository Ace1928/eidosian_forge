import cudf
import cupy
import cupy as cp
import numpy as np
import ray
from pandas.core.dtypes.common import is_list_like
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.ray.common import RayWrapper
from modin.core.execution.ray.common.utils import ObjectIDType
def get_gpu_manager(self):
    """
        Get gpu manager associated with this partition.

        Returns
        -------
        modin.core.execution.ray.implementations.cudf_on_ray.partitioning.GPUManager
            ``GPUManager`` associated with this object.
        """
    return self.gpu_manager