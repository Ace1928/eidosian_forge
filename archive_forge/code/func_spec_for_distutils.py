import sys
import os
def spec_for_distutils(self):
    if self.is_cpython():
        return None
    import importlib
    import importlib.abc
    import importlib.util
    try:
        mod = importlib.import_module('setuptools._distutils')
    except Exception:
        return None

    class DistutilsLoader(importlib.abc.Loader):

        def create_module(self, spec):
            mod.__name__ = 'distutils'
            return mod

        def exec_module(self, module):
            pass
    return importlib.util.spec_from_loader('distutils', DistutilsLoader(), origin=mod.__file__)