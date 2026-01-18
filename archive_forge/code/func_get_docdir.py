import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from . import util
def get_docdir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'doc', 'source', 'f2py', 'code'))