import contextlib
import os
import re
import sys
from distutils.core import Command
from distutils.errors import *
from distutils.sysconfig import customize_compiler, get_python_version
from distutils.sysconfig import get_config_h_filename
from distutils.dep_util import newer_group
from distutils.extension import Extension
from distutils.util import get_platform
from distutils import log
from site import USER_BASE
def get_export_symbols(self, ext):
    """Return the list of symbols that a shared extension has to
        export.  This either uses 'ext.export_symbols' or, if it's not
        provided, "PyInit_" + module_name.  Only relevant on Windows, where
        the .pyd file (DLL) must export the module "PyInit_" function.
        """
    suffix = '_' + ext.name.split('.')[-1]
    try:
        suffix.encode('ascii')
    except UnicodeEncodeError:
        suffix = 'U' + suffix.encode('punycode').replace(b'-', b'_').decode('ascii')
    initfunc_name = 'PyInit' + suffix
    if initfunc_name not in ext.export_symbols:
        ext.export_symbols.append(initfunc_name)
    return ext.export_symbols