import abc
import collections.abc
from collections import OrderedDict
import enum
import itertools
import typing
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
from rpy2.rinterface_lib._rinterface_capi import _evaluated_promise
from rpy2.rinterface_lib._rinterface_capi import SupportsSEXP
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
from rpy2.rinterface_lib import na_values
def rsame(self, sexp) -> bool:
    if isinstance(sexp, Sexp):
        return self.__sexp__._cdata == sexp.__sexp__._cdata
    elif isinstance(sexp, _rinterface.SexpCapsule):
        return sexp._cdata == sexp._cdata
    else:
        raise ValueError('Not an R object.')