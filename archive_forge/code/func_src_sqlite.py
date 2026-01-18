from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def src_sqlite(*args, **kwargs):
    res = dplyr.src_sqlite(*args, **kwargs)
    return DataSource(res)