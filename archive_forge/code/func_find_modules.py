from distutils.command.build_py import build_py as old_build_py
from numpy.distutils.misc_util import is_string
def find_modules(self):
    old_py_modules = self.py_modules[:]
    new_py_modules = [_m for _m in self.py_modules if is_string(_m)]
    self.py_modules[:] = new_py_modules
    modules = old_build_py.find_modules(self)
    self.py_modules[:] = old_py_modules
    return modules