import cudf
import pandas
import ray
from modin.core.execution.ray.common import RayWrapper
def get_oid(self, key):
    """
        Get the value from `self.cudf_dataframe_dict` by `key`.

        Parameters
        ----------
        key : int
            The key to get value.

        Returns
        -------
        cudf.DataFrame
            Dataframe corresponding to `key`(will be a ``ray.ObjectRef``
            in outside level).
        """
    return self.cudf_dataframe_dict[key]