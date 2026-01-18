from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def slice_min(self, *args, **kwargs):
    """Call the R function `dplyr::slice_min()`."""
    res = dplyr.slice_min(self, *args, **kwargs)
    return res