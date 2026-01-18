import abc
import os
import typing
import warnings
import weakref
import rpy2.rinterface
import rpy2.rinterface_lib.callbacks
from rpy2.robjects import conversion
@rclass.setter
def rclass(self, value):
    if isinstance(value, str):
        value = (value,)
    new_cls = rpy2.rinterface.StrSexpVector(value)
    rpy2.rinterface.sexp.rclass_set(self.__sexp__, new_cls)