import abc
import collections.abc
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import sexp
from . import conversion
import rpy2.rlike.container as rlc
import datetime
import copy
import itertools
import math
import os
import jinja2  # type: ignore
import time
import tzlocal
from time import struct_time, mktime
import typing
import warnings
from rpy2.rinterface import (Sexp, ListSexpVector, StrSexpVector,
def cbind(self, *args, **kwargs):
    """ bind objects as supplementary columns """
    new_args = [self] + [conversion.rpy2py(x) for x in args]
    new_kwargs = dict([(k, conversion.rpy2py(v)) for k, v in kwargs.items()])
    res = self._cbind(*new_args, **new_kwargs)
    return conversion.get_conversion().rpy2py(res)