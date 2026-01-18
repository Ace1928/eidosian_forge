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
def tabulate(self, nbins=None):
    """ Like the R function tabulate,
        count the number of times integer values are found """
    if nbins is None:
        nbins = max(1, max(self))
    res = self._tabulate(self)
    return conversion.rpy2py(res)