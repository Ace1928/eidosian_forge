from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def mutate_all(self, *args, **kwargs):
    """Call the R function `dplyr::mutate_all()`."""
    res = dplyr.mutate_all(self, *args, **kwargs)
    return res