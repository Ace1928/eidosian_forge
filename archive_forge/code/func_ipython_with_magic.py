import pytest
import textwrap
import types
import warnings
from itertools import product
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface_lib._rinterface_capi
import rpy2.robjects
import rpy2.robjects.conversion
from .. import utils
from io import StringIO
from rpy2 import rinterface
from rpy2.robjects import r, vectors, globalenv
import rpy2.robjects.packages as rpacks
@pytest.fixture(scope='module')
def ipython_with_magic():
    if IPython is None:
        return None
    ip = get_ipython()
    ip.run_line_magic('load_ext', 'rpy2.ipython')
    return ip