import sys
import os
def ensure_local_distutils():
    import importlib
    clear_distutils()
    with shim():
        importlib.import_module('distutils')
    core = importlib.import_module('distutils.core')
    assert '_distutils' in core.__file__, core.__file__
    assert 'setuptools._distutils.log' not in sys.modules