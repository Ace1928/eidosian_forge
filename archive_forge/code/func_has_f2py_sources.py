import re
from distutils.extension import Extension as old_Extension
def has_f2py_sources(self):
    for source in self.sources:
        if fortran_pyf_ext_re(source):
            return True
    return False