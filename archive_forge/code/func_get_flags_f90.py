import os
from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32
def get_flags_f90(self):
    opt = FCompiler.get_flags_f90(self)
    opt.extend(['-YCFRL=1', '-YCOM_NAMES=LCS', '-YCOM_PFX', '-YEXT_PFX', '-YCOM_SFX=_', '-YEXT_SFX=_', '-YEXT_NAMES=LCS'])
    if self.get_version():
        if self.get_version() > '4.6':
            opt.extend(['-YDEALLOC=ALL'])
    return opt