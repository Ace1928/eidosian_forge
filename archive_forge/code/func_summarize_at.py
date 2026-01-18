from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def summarize_at(self, *args, **kwargs):
    """Call the R function `dplyr::summarize_at()`.

        This can return a GroupedDataFrame or a DataFrame.
        """
    res = dplyr.summarize_at(self, *args, **kwargs)
    return guess_wrap_type(res)(res)