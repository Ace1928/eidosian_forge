import sys
from functools import partial
from pydev_ipython.version import check_version
def forbid(self, module_name):
    sys.modules.pop(module_name, None)
    self.__forbidden.add(module_name)