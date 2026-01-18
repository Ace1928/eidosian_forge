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
def iter_localized_datetime(self):
    """Iterator yielding localized Python datetime objects."""
    try:
        r_tzone_name = self.do_slot('tzone')[0]
    except LookupError:
        warnings.warn('R object inheriting from "POSIXct" but without attribute "tzone".')
        r_tzone_name = ''
    if r_tzone_name == '':
        r_tzone = get_timezone()
    else:
        r_tzone = zoneinfo.ZoneInfo(r_tzone_name)
    for x in self:
        yield (None if math.isnan(x) else POSIXct._datetime_from_timestamp(x, r_tzone))