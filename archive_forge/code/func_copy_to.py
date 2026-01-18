from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def copy_to(self, destination, name, **kwargs):
    """
        - destination: database
        - name: table name in the destination database
        """
    res = dplyr.copy_to(destination, self, name=name)
    return guess_wrap_type(res)(res)