import abc
import enum
import inspect
import logging
from typing import Tuple
import typing
import warnings
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from _cffi_backend import FFI  # type: ignore
def build_rcall(rfunction, args=[], kwargs=[]):
    rlib = openrlib.rlib
    with memorymanagement.rmemory() as rmemory:
        rcall = rmemory.protect(rlib.Rf_allocList(len(args) + len(kwargs) + 1))
        _SET_TYPEOF(rcall, rlib.LANGSXP)
        rlib.SETCAR(rcall, rfunction)
        item = rlib.CDR(rcall)
        for val in args:
            cdata = rmemory.protect(conversion._get_cdata(val))
            rlib.SETCAR(item, cdata)
            item = rlib.CDR(item)
        for key, val in kwargs:
            if key is not None:
                _assert_valid_slotname(key)
                rlib.SET_TAG(item, rlib.Rf_install(conversion._str_to_cchar(key)))
            cdata = rmemory.protect(conversion._get_cdata(val))
            rlib.SETCAR(item, cdata)
            item = rlib.CDR(item)
    return rcall