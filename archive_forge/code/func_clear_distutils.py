import sys
import os
def clear_distutils():
    if 'distutils' not in sys.modules:
        return
    import warnings
    warnings.warn('Setuptools is replacing distutils.')
    mods = [name for name in sys.modules if name == 'distutils' or name.startswith('distutils.')]
    for name in mods:
        del sys.modules[name]