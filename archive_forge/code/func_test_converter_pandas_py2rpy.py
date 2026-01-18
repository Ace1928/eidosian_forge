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
@pytest.mark.parametrize('python_obj_code,r_cls', (("\npd.DataFrame.from_dict(\n  {\n    'x': (1, 2, 3),\n    'y': (2.9, 3.5, 2.1),\n    'z': ('a', 'b', 'c')\n  }\n)", ('data.frame',)), ("\npd.Categorical.from_codes(\n  [0, 1, 0],\n  categories=['a', 'b'],\n  ordered=False\n)", ('factor',))))
def test_converter_pandas_py2rpy(ipython_with_magic, clean_globalenv, python_obj_code, r_cls):
    py_obj = eval(python_obj_code)
    ipython_with_magic.user_ns['py_obj'] = py_obj
    ipython_with_magic.run_line_magic('Rpush', 'py_obj')
    assert tuple(rpy2.robjects.r('class(py_obj)')) == r_cls