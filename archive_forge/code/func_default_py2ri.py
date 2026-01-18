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
def default_py2ri(o):
    """ Convert an arbitrary Python object to a
    :class:`rpy2.rinterface.Sexp` object.
    Creates an R object with the content of the Python object,
    wich means data copying.
    :param o: object
    :rtype: :class:`rpy2.rinterface.Sexp` (and subclasses)
    """
    pass