from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def slice_head(self, *args, **kwargs):
    """Call the R function `dplyr::slice_head()`."""
    res = dplyr.slice_head(self, *args, **kwargs)
    return res