import os
import re
import sys
import platform
import shlex
import time
import subprocess
from copy import copy
from pathlib import Path
from distutils import ccompiler
from distutils.ccompiler import (
from distutils.errors import (
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion
from numpy.distutils import log
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
import threading
def new_compiler(plat=None, compiler=None, verbose=None, dry_run=0, force=0):
    if verbose is None:
        verbose = log.get_threshold() <= log.INFO
    if plat is None:
        plat = os.name
    try:
        if compiler is None:
            compiler = get_default_compiler(plat)
        module_name, class_name, long_description = compiler_class[compiler]
    except KeyError:
        msg = "don't know how to compile C/C++ code on platform '%s'" % plat
        if compiler is not None:
            msg = msg + " with '%s' compiler" % compiler
        raise DistutilsPlatformError(msg)
    module_name = 'numpy.distutils.' + module_name
    try:
        __import__(module_name)
    except ImportError as e:
        msg = str(e)
        log.info('%s in numpy.distutils; trying from distutils', str(msg))
        module_name = module_name[6:]
        try:
            __import__(module_name)
        except ImportError as e:
            msg = str(e)
            raise DistutilsModuleError("can't compile C/C++ code: unable to load module '%s'" % module_name)
    try:
        module = sys.modules[module_name]
        klass = vars(module)[class_name]
    except KeyError:
        raise DistutilsModuleError(("can't compile C/C++ code: unable to find class '%s' " + "in module '%s'") % (class_name, module_name))
    compiler = klass(None, dry_run, force)
    compiler.verbose = verbose
    log.debug('new_compiler returns %s' % klass)
    return compiler