from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def left_join(self, *args, **kwargs):
    """Call the R function `dplyr::left_join()`."""
    res = dplyr.left_join(self, *args, **kwargs)
    return res