import os
from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32
def get_libraries(self):
    opt = FCompiler.get_libraries(self)
    if self.get_version() >= '11.0':
        opt.extend(['af90math', 'afio', 'af77math', 'amisc'])
    elif self.get_version() >= '10.0':
        opt.extend(['af90math', 'afio', 'af77math', 'U77'])
    elif self.get_version() >= '8.0':
        opt.extend(['f90math', 'fio', 'f77math', 'U77'])
    else:
        opt.extend(['fio', 'f90math', 'fmath', 'U77'])
    if os.name == 'nt':
        opt.append('COMDLG32')
    return opt