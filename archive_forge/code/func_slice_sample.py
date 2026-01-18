from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def slice_sample(self, *args, **kwargs):
    """Call the R function `dplyr::slice_sample()`."""
    res = dplyr.slice_sample(self, *args, **kwargs)
    return res