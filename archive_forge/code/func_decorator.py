import enum
import logging
import os
import types
import typing
def decorator(func: typing.Callable) -> object:
    if get_ffi_mode(_rinterface_cffi) == InterfaceType.ABI:
        res = _rinterface_cffi.ffi.callback(definition.callback_def)(func)
    elif get_ffi_mode(_rinterface_cffi) == InterfaceType.API:
        res = _rinterface_cffi.ffi.def_extern()(func)
    else:
        raise RuntimeError('The cffi mode is neither ABI or API.')
    return res