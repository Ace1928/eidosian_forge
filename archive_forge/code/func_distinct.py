from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def distinct(self, *args, _keep_all=False):
    """Call the R function `dplyr::distinct()`."""
    res = dplyr.distinct(self, *args, **{'.keep_all': _keep_all})
    return res