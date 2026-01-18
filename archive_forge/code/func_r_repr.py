import abc
import os
import typing
import warnings
import weakref
import rpy2.rinterface
import rpy2.rinterface_lib.callbacks
from rpy2.robjects import conversion
def r_repr(self):
    """ String representation for an object that can be
        directly evaluated as R code.
        """
    return repr_robject(self, linesep='\n')