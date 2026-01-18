from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def right_join(self, *args, **kwargs):
    """Call the R function `dplyr::right_join()`."""
    res = dplyr.right_join(self, *args, **kwargs)
    return res