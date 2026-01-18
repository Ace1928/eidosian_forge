import dill
from functools import partial
import warnings
def test_function_cells():
    assert copy(f())