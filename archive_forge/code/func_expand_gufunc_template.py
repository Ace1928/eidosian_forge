from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
def expand_gufunc_template(template, indims, outdims, funcname, argtypes):
    """Expand gufunc source template
    """
    argdims = indims + outdims
    argnames = ['arg{0}'.format(i) for i in range(len(argdims))]
    checkedarg = 'min({0})'.format(', '.join(['{0}.shape[0]'.format(a) for a in argnames]))
    inputs = [_gen_src_for_indexing(aref, adims, atype) for aref, adims, atype in zip(argnames, indims, argtypes)]
    outputs = [_gen_src_for_indexing(aref, adims, atype) for aref, adims, atype in zip(argnames[len(indims):], outdims, argtypes[len(indims):])]
    argitems = inputs + outputs
    src = template.format(name=funcname, args=', '.join(argnames), checkedarg=checkedarg, argitems=', '.join(argitems))
    return src