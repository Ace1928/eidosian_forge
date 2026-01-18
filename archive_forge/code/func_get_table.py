from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def get_table(self, name):
    """ "Get" table from a source (R dplyr's function `tbl`). """
    return DataFrame(tbl(self, name))