from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@property
def tablenames(self):
    """ Call the R function dplyr::src_tbls() and return a vector
        of table names."""
    return tuple(dplyr.src_tbls(self))