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
def repr_format_elt(self, elt, max_width=12):
    max_width = int(max_width)
    str_elt = str(elt)
    if len(str_elt) < max_width:
        res = elt
    else:
        res = '%s...' % str_elt[:max_width - 3]
    return res