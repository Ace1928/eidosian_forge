from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@property
def is_grouped_df(self) -> bool:
    """Is the DataFrame in a grouped state"""
    return dplyr.is_grouped_df(self)[0]