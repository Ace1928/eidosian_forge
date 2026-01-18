import array
import contextlib
import os
import types
import typing
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.embedded
import rpy2.rinterface_lib.openrlib
import rpy2.rlike.container as rlc
from rpy2.robjects.robject import RObjectMixin, RObject
import rpy2.robjects.functions
from rpy2.robjects.environments import (Environment,
from rpy2.robjects.methods import methods_env
from rpy2.robjects.methods import RS4
from . import conversion
from . import vectors
from . import language
from rpy2.rinterface import (Sexp,
from rpy2.robjects.functions import Function
from rpy2.robjects.functions import SignatureTranslatedFunction
def setenvironment(self, val):
    """ Set the environment in which a formula will find its symbols."""
    if not isinstance(val, rinterface.SexpEnvironment):
        raise TypeError('The environment must be an instance of rpy2.rinterface.Sexp.environment')
    self.do_slot_assign('.Environment', val)