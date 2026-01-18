import os
from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32
def library_option(self, lib):
    if os.name == 'nt':
        return '%s.lib' % lib
    return '-l' + lib