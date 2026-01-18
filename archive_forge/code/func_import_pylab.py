from io import BytesIO
from binascii import b2a_base64
from functools import partial
import warnings
from IPython.core.display import _pngxy
from IPython.utils.decorators import flag_calls
def import_pylab(user_ns, import_all=True):
    """Populate the namespace with pylab-related values.

    Imports matplotlib, pylab, numpy, and everything from pylab and numpy.

    Also imports a few names from IPython (figsize, display, getfigs)

    """
    s = 'import numpy\nimport matplotlib\nfrom matplotlib import pylab, mlab, pyplot\nnp = numpy\nplt = pyplot\n'
    exec(s, user_ns)
    if import_all:
        s = 'from matplotlib.pylab import *\nfrom numpy import *\n'
        exec(s, user_ns)
    user_ns['figsize'] = figsize
    from IPython.display import display
    user_ns['display'] = display
    user_ns['getfigs'] = getfigs