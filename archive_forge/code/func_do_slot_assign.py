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
def do_slot_assign(self, name: str, value) -> None:
    _rinterface._assert_valid_slotname(name)
    cchar = conversion._str_to_cchar(name)
    with memorymanagement.rmemory() as rmemory:
        name_r = rmemory.protect(openrlib.rlib.Rf_install(cchar))
        cdata = rmemory.protect(conversion._get_cdata(value))
        openrlib.rlib.R_do_slot_assign(self.__sexp__._cdata, name_r, cdata)