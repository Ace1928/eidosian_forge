from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def sample_n(self, *args):
    """Call the R function `dplyr::sample_n()`."""
    res = dplyr.sample_n(self, *args)
    return res