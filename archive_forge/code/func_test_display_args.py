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
def test_display_args(ipython_with_magic, clean_globalenv):
    cell = '\n    x <- 123\n    as.integer(x + 1)\n    '
    res = []

    def display(x, a):
        res.append(x)
    with pytest.raises(NameError):
        ipython_with_magic.run_cell_magic('R', '--display=mydisplay', cell)
    ipython_with_magic.push({'mydisplay': display})
    ipython_with_magic.run_cell_magic('R', '--display=mydisplay', cell)
    assert len(res) == 1
    assert tuple(res[0]) == (124,)