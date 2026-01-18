import abc
import atexit
import contextlib
import contextvars
import csv
import enum
import functools
import inspect
import os
import math
import platform
import signal
import subprocess
import textwrap
import threading
import typing
import warnings
from typing import Union
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
import rpy2.rinterface_lib.embedded as embedded
import rpy2.rinterface_lib.conversion as conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
import rpy2.rinterface_lib.memorymanagement as memorymanagement
from rpy2.rinterface_lib import na_values
from rpy2.rinterface_lib.sexp import NULL
from rpy2.rinterface_lib.sexp import NULLType
import rpy2.rinterface_lib.bufferprotocol as bufferprotocol
from rpy2.rinterface_lib import sexp
from rpy2.rinterface_lib.sexp import CharSexp  # noqa: F401
from rpy2.rinterface_lib.sexp import RTYPES
from rpy2.rinterface_lib.sexp import SexpVector
from rpy2.rinterface_lib.sexp import StrSexpVector
from rpy2.rinterface_lib.sexp import Sexp
from rpy2.rinterface_lib.sexp import SexpEnvironment
from rpy2.rinterface_lib.sexp import unserialize  # noqa: F401
from rpy2.rinterface_lib.sexp import emptyenv
from rpy2.rinterface_lib.sexp import baseenv
from rpy2.rinterface_lib.sexp import globalenv
def evalr_expr_with_visible(expr: 'ExprSexpVector', envir: typing.Union[None, 'SexpEnvironment']=None) -> 'ListSexpVector':
    """Evaluate an R expression and return value and visibility flag.

    :param expr: An R expression.
    :param envir: An environment in which the expression will be evaluated.

    :return: An R list with (value, visibility) where visibility is
    an R boolean.
    """
    if envir is None:
        envir = evaluation_context.get()
    assert isinstance(envir, SexpEnvironment)
    error_occured = _rinterface.ffi.new('int *', 0)
    with memorymanagement.rmemory() as rmemory:
        r_call_nested = rmemory.protect(openrlib.rlib.Rf_lang2(baseenv['eval'].__sexp__._cdata, expr.__sexp__._cdata))
        r_call = rmemory.protect(openrlib.rlib.Rf_lang2(baseenv['withVisible'].__sexp__._cdata, r_call_nested))
        r_res = rmemory.protect(openrlib.rlib.R_tryEval(r_call, envir.__sexp__._cdata, error_occured))
        if error_occured[0]:
            raise embedded.RRuntimeError(_rinterface._geterrmessage())
        res = conversion._cdata_to_rinterface(r_res)
        assert isinstance(res, ListSexpVector)
    return res