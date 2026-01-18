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
def serialize(cdata: FFI.CData, cdata_env: FFI.CData) -> FFI.CData:
    """Serialize an R object to an R string.

    Note that the R string returned is *not* protected from
    the R garbage collection."""
    rlib = openrlib.rlib
    with memorymanagement.rmemory() as rmemory:
        sym_serialize = rmemory.protect(rlib.Rf_install(conversion._str_to_cchar('serialize')))
        func_serialize = rmemory.protect(_findfun(sym_serialize, rlib.R_BaseEnv))
        r_call = rmemory.protect(rlib.Rf_lang3(func_serialize, cdata, rlib.R_NilValue))
        error_occured = ffi.new('int *', 0)
        res = rlib.R_tryEval(r_call, cdata_env, error_occured)
        if error_occured[0]:
            raise embedded.RRuntimeError(_geterrmessage())
        return res