import contextlib
import os
import typing
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp as sexp
from rpy2.robjects.robject import RObjectMixin
from rpy2.robjects import conversion
@enclos.setter
def enclos(self, value: sexp.SexpEnvironment) -> None:
    super().enclos = value