import cudf
import pandas
import ray
from modin.core.execution.ray.common import RayWrapper
def store_new_df(self, df):
    """
        Store `df` in `self.cudf_dataframe_dict`.

        Parameters
        ----------
        df : cudf.DataFrame
            The ``cudf.DataFrame`` to be added.

        Returns
        -------
        int
            The key associated with added dataframe
            (will be a ``ray.ObjectRef`` in outside level).
        """
    self.key += 1
    self.cudf_dataframe_dict[self.key] = df
    return self.key