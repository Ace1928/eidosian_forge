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
@pytest.mark.skipif(IPython is None, reason='The optional package IPython cannot be imported.')
@pytest.mark.skipif(not has_pandas, reason='pandas not installed')
@pytest.mark.parametrize('r_obj_code,py_cls', (("\ndata.frame(\n  x = c(1, 2, 3),\n  y = c(2.9, 3.5, 2.1),\n  z = c('a', 'b', 'c')\n)", 'pd.DataFrame'), ("\nfactor(c('a', 'b', 'a'),\n       ordered = TRUE)\n", 'pd.Categorical')))
def test_converter_pandas_rpy2py(ipython_with_magic, clean_globalenv, r_obj_code, py_cls):
    rpy2.robjects.r(f'r_obj <- {r_obj_code}')
    py_obj = ipython_with_magic.run_line_magic('Rget', 'r_obj')
    assert isinstance(py_obj, eval(py_cls))