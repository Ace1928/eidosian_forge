import os
from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32
def get_flags_f77(self):
    opt = FCompiler.get_flags_f77(self)
    opt.extend(['-N22', '-N90', '-N110'])
    v = self.get_version()
    if os.name == 'nt':
        if v and v >= '8.0':
            opt.extend(['-f', '-N15'])
    else:
        opt.append('-f')
        if v:
            if v <= '4.6':
                opt.append('-B108')
            else:
                opt.append('-N15')
    return opt