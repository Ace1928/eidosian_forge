import os
from distutils.dist import Distribution
def dump_variables(self):
    for name in self._conf_keys:
        self.dump_variable(name)