from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def src_mysql(*args, **kwargs):
    res = dplyr.src_mysql(*args, **kwargs)
    return DataSource(res)