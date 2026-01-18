import os
import sys
import subprocess
import re
import textwrap
import numpy.distutils.ccompiler  # noqa: F401
from numpy.distutils import log
import distutils.cygwinccompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import get_build_version as get_build_msvc_version
from distutils.errors import UnknownFileError
from numpy.distutils.misc_util import (msvc_runtime_library,
def generate_def(dll, dfile):
    """Given a dll file location,  get all its exported symbols and dump them
    into the given def file.

    The .def file will be overwritten"""
    dump = dump_table(dll)
    for i in range(len(dump)):
        if _START.match(dump[i].decode()):
            break
    else:
        raise ValueError('Symbol table not found')
    syms = []
    for j in range(i + 1, len(dump)):
        m = _TABLE.match(dump[j].decode())
        if m:
            syms.append((int(m.group(1).strip()), m.group(2)))
        else:
            break
    if len(syms) == 0:
        log.warn('No symbols found in %s' % dll)
    with open(dfile, 'w') as d:
        d.write('LIBRARY        %s\n' % os.path.basename(dll))
        d.write(';CODE          PRELOAD MOVEABLE DISCARDABLE\n')
        d.write(';DATA          PRELOAD SINGLE\n')
        d.write('\nEXPORTS\n')
        for s in syms:
            d.write('%s\n' % s[1])