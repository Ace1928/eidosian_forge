import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
@pytest.mark.skipif(sys.flags.optimize > 1, reason='no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1')
def test_doc_note():

    def method(self):
        """This docstring

        Has multiple lines

        And notes

        Notes
        -----
        original note
        """
        pass
    expected_doc = 'This docstring\n\nHas multiple lines\n\nAnd notes\n\nNotes\n-----\nnote\n\noriginal note'
    assert_equal(np.ma.core.doc_note(method.__doc__, 'note'), expected_doc)