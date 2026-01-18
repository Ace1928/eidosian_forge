import os
import sys
from os.path import pardir, realpath
def is_python_build(check_home=None):
    if check_home is not None:
        import warnings
        warnings.warn('check_home argument is deprecated and ignored.', DeprecationWarning, stacklevel=2)
    for fn in ('Setup', 'Setup.local'):
        if os.path.isfile(os.path.join(_PROJECT_BASE, 'Modules', fn)):
            return True
    return False