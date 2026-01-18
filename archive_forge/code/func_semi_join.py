from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def semi_join(self, *args, **kwargs):
    """Call the R function `dplyr::semi_join()`."""
    res = dplyr.semi_join(self, *args, **kwargs)
    return res