import os
import pathlib
import subprocess
import sys
import sysconfig
import textwrap
def build_and_import_extension(modname, functions, *, prologue='', build_dir=None, include_dirs=[], more_init=''):
    '''
    Build and imports a c-extension module `modname` from a list of function
    fragments `functions`.


    Parameters
    ----------
    functions : list of fragments
        Each fragment is a sequence of func_name, calling convention, snippet.
    prologue : string
        Code to precede the rest, usually extra ``#include`` or ``#define``
        macros.
    build_dir : pathlib.Path
        Where to build the module, usually a temporary directory
    include_dirs : list
        Extra directories to find include files when compiling
    more_init : string
        Code to appear in the module PyMODINIT_FUNC

    Returns
    -------
    out: module
        The module will have been loaded and is ready for use

    Examples
    --------
    >>> functions = [("test_bytes", "METH_O", """
        if ( !PyBytesCheck(args)) {
            Py_RETURN_FALSE;
        }
        Py_RETURN_TRUE;
    """)]
    >>> mod = build_and_import_extension("testme", functions)
    >>> assert not mod.test_bytes(u'abc')
    >>> assert mod.test_bytes(b'abc')
    '''
    body = prologue + _make_methods(functions, modname)
    init = 'PyObject *mod = PyModule_Create(&moduledef);\n           '
    if not build_dir:
        build_dir = pathlib.Path('.')
    if more_init:
        init += '#define INITERROR return NULL\n                '
        init += more_init
    init += '\nreturn mod;'
    source_string = _make_source(modname, init, body)
    try:
        mod_so = compile_extension_module(modname, build_dir, include_dirs, source_string)
    except Exception as e:
        raise RuntimeError(f'could not compile in {build_dir}:') from e
    import importlib.util
    spec = importlib.util.spec_from_file_location(modname, mod_so)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo