import os
from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32
def get_flags(self):
    opt = FCompiler.get_flags(self)
    if os.name != 'nt':
        opt.extend(['-s'])
        if self.get_version():
            if self.get_version() >= '8.2':
                opt.append('-fpic')
    return opt