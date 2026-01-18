from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def select(self, *args):
    """Call the R function `dplyr::select()`."""
    res = dplyr.select(self, *args)
    return res