import cffi  # type: ignore
import os
import re
import sys
import warnings
import situation  # preloaded in setup.py
import importlib
def createbuilder_api():
    ffibuilder = cffi.FFI()
    definitions = {}
    define_rlen_kind(ffibuilder, definitions)
    define_osname(definitions)
    if not os.name == 'nt':
        definitions['R_INTERFACE_PTRS'] = True
    r_h = read_source('R_API.h')
    eventloop_h = read_source('R_API_eventloop.h')
    eventloop_c = read_source('R_API_eventloop.c')
    rpy2_h = read_source('RPY2.h')
    r_home = situation.get_r_home()
    if r_home is None:
        sys.exit('Error: rpy2 in API mode cannot be built without R in the PATH or R_HOME defined. Correct this or force ABI mode-only by defining the environment variable RPY2_CFFI_MODE=ABI')
    c_ext = situation.CExtensionOptions()
    c_ext.add_lib(*situation.get_r_flags(r_home, '--ldflags'))
    c_ext.add_lib(*situation.get_r_libs(r_home, 'BLAS_LIBS'))
    c_ext.add_include(*situation.get_r_flags(r_home, '--cppflags'))
    c_ext.extra_link_args.append(f'-Wl,-rpath,{situation.get_rlib_rpath(r_home)}')
    if 'RPY2_RLEN_LONG' in definitions:
        definitions['RPY2_RLEN_LONG'] = True
    ffibuilder.set_source('_rinterface_cffi_api', eventloop_c + rpy2_h, libraries=c_ext.libraries, library_dirs=c_ext.library_dirs, include_dirs=['rpy2/rinterface_lib/'], define_macros=list(definitions.items()), extra_compile_args=c_ext.extra_compile_args, extra_link_args=c_ext.extra_link_args)
    callback_defns_api = '\n'.join((x.extern_python_def for x in [ffi_proxy._capsule_finalizer_def, ffi_proxy._evaluate_in_r_def, ffi_proxy._consoleflush_def, ffi_proxy._consoleread_def, ffi_proxy._consolereset_def, ffi_proxy._consolewrite_def, ffi_proxy._consolewrite_ex_def, ffi_proxy._showmessage_def, ffi_proxy._choosefile_def, ffi_proxy._cleanup_def, ffi_proxy._showfiles_def, ffi_proxy._processevents_def, ffi_proxy._busy_def, ffi_proxy._callback_def, ffi_proxy._yesnocancel_def, ffi_proxy._parsevector_wrap_def, ffi_proxy._handler_def, ffi_proxy._exec_findvar_in_frame_def]))
    cdef_r, _ = c_preprocess(iter(r_h.split('\n')), definitions=definitions, rownum=1)
    ffibuilder.cdef('\n'.join(cdef_r))
    ffibuilder.cdef(rpy2_h)
    ffibuilder.cdef(callback_defns_api)
    cdef_eventloop, _ = c_preprocess(iter(eventloop_h.split('\n')), definitions={'CFFI_SOURCE': True}, rownum=1)
    ffibuilder.cdef('\n'.join(cdef_eventloop))
    ffibuilder.cdef('void rpy2_runHandlers(InputHandler *handlers);')
    return ffibuilder