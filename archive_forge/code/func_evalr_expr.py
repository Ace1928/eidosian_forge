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
def evalr_expr(expr: 'ExprSexpVector', envir: typing.Union[None, 'SexpEnvironment', 'NULLType', 'ListSexpVector', 'PairlistSexpVector', int, '_MissingArgType']=None, enclos: typing.Union[None, 'ListSexpVector', 'PairlistSexpVector', 'NULLType', '_MissingArgType']=None) -> sexp.Sexp:
    """Evaluate an R expression.

    :param expr: An R expression.
    :param envir: An optional R environment in which the evaluation
      will happen. If None, which is the default, the environment in
     the context variable `evaluation_context` is used.
    :param enclos: An enclosure. This is only relevant when envir
      is a list, a pairlist, or a data.frame. It specifies where to
    look for objects not found in envir.
    :return: The R objects resulting from the evaluation of the code"""
    if envir is None:
        envir = evaluation_context.get()
    if enclos is None:
        enclos = MissingArg
    res = baseenv['eval'](expr, envir=envir, enclos=enclos)
    return res