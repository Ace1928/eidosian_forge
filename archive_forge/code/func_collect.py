from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def collect(self, *args, **kwargs):
    """Call the function `collect` in the R package `dplyr`."""
    return dplyr.collect(self, *args, **kwargs)