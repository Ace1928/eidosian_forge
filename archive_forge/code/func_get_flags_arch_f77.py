from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler
def get_flags_arch_f77(self):
    r = None
    if cpu.is_r10000():
        r = 10000
    elif cpu.is_r12000():
        r = 12000
    elif cpu.is_r8000():
        r = 8000
    elif cpu.is_r5000():
        r = 5000
    elif cpu.is_r4000():
        r = 4000
    if r is not None:
        return ['r%s' % r]
    return []