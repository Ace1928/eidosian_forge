from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler
def get_flags_arch_f90(self):
    r = self.get_flags_arch_f77()
    if r:
        r[0] = '-' + r[0]
    return r