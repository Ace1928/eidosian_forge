from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def summarize(self, *args, **kwargs):
    """Call the R function `dplyr::summarize()`.

        This can return a GroupedDataFrame or a DataFrame.
        """
    res = dplyr.summarize(self, *args, **kwargs)
    return guess_wrap_type(res)(res)