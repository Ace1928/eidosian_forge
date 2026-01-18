import sys
import unittest
from test.support.os_helper import EnvironmentVarGuard
from distutils import sysconfig
from distutils.unixccompiler import UnixCCompiler
def gcv(v):
    if v == 'LDSHARED':
        return 'gcc-4.2 -bundle -undefined dynamic_lookup '
    return 'gcc-4.2'