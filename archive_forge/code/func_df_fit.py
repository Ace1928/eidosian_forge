import numpy as np
from . import kernels
def df_fit(self):
    """alias of df_model for backwards compatibility
        """
    return self.df_model()