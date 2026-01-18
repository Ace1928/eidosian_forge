import os
import sys
def get_pydevd_file():
    import pydevd
    f = pydevd.__file__
    if f.endswith('.pyc'):
        f = f[:-1]
    elif f.endswith('$py.class'):
        f = f[:-len('$py.class')] + '.py'
    return f