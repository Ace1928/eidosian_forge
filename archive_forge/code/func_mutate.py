from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def mutate(self, **kwargs):
    """Call the R function `dplyr::mutate()`."""
    res = dplyr.mutate(self, **kwargs)
    return res