from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def inner_join(self, *args, **kwargs):
    """Call the R function `dplyr::inner_join()`."""
    res = dplyr.inner_join(self, *args, **kwargs)
    return res