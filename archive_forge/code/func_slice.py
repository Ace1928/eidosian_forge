from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def slice(self, *args, **kwargs):
    """Call the R function `dplyr::slice()`."""
    res = dplyr.slice(self, *args, **kwargs)
    return res