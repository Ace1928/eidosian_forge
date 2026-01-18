import enum
import logging
import os
import types
import typing
def get_ffi_mode(_rinterface_cffi: types.ModuleType) -> InterfaceType:
    global FFI_MODE
    if FFI_MODE is None:
        if hasattr(_rinterface_cffi, 'lib'):
            res = InterfaceType.API
        else:
            res = InterfaceType.ABI
        logger.debug(f'cffi mode is {res}')
        FFI_MODE = res
    return FFI_MODE