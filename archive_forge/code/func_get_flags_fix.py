import os
from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32
def get_flags_fix(self):
    opt = FCompiler.get_flags_fix(self)
    opt.extend(['-YCFRL=1', '-YCOM_NAMES=LCS', '-YCOM_PFX', '-YEXT_PFX', '-YCOM_SFX=_', '-YEXT_SFX=_', '-YEXT_NAMES=LCS'])
    opt.extend(['-f', 'fixed'])
    return opt