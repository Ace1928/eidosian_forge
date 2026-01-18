import os
import sys
import copy
from subprocess import Popen, PIPE, check_output
import re
from distutils.unixccompiler import UnixCCompiler
from distutils.file_util import write_file
from distutils.errors import (DistutilsExecError, CCompilerError,
from distutils.version import LooseVersion
from distutils.spawn import find_executable
def check_config_h():
    """Check if the current Python installation appears amenable to building
    extensions with GCC.

    Returns a tuple (status, details), where 'status' is one of the following
    constants:

    - CONFIG_H_OK: all is well, go ahead and compile
    - CONFIG_H_NOTOK: doesn't look good
    - CONFIG_H_UNCERTAIN: not sure -- unable to read pyconfig.h

    'details' is a human-readable string explaining the situation.

    Note there are two ways to conclude "OK": either 'sys.version' contains
    the string "GCC" (implying that this Python was built with GCC), or the
    installed "pyconfig.h" contains the string "__GNUC__".
    """
    from distutils import sysconfig
    if 'GCC' in sys.version:
        return (CONFIG_H_OK, "sys.version mentions 'GCC'")
    fn = sysconfig.get_config_h_filename()
    try:
        config_h = open(fn)
        try:
            if '__GNUC__' in config_h.read():
                return (CONFIG_H_OK, "'%s' mentions '__GNUC__'" % fn)
            else:
                return (CONFIG_H_NOTOK, "'%s' does not mention '__GNUC__'" % fn)
        finally:
            config_h.close()
    except OSError as exc:
        return (CONFIG_H_UNCERTAIN, "couldn't read '%s': %s" % (fn, exc.strerror))