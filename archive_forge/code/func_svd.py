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
def svd(self, nu=None, nv=None, linpack=False):
    """ SVD decomposition.
        If nu is None, it is given the default value min(tuple(self.dim)).
        If nv is None, it is given the default value min(tuple(self.dim)).
        """
    if nu is None:
        nu = min(tuple(self.dim))
    if nv is None:
        nv = min(tuple(self.dim))
    res = self._svd(self, nu=nu, nv=nv)
    return conversion.get_conversion().rpy2py(res)