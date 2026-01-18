import abc
import os
import typing
import warnings
import weakref
import rpy2.rinterface
import rpy2.rinterface_lib.callbacks
from rpy2.robjects import conversion
def repr_robject(o, linesep=os.linesep):
    s = rpy2.rinterface.baseenv.find('deparse')(o)
    s = str.join(linesep, s)
    return s