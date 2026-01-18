import dill
from functools import partial
import warnings
def test_circular_reference():
    assert copy(obj4())
    obj4_copy = dill.loads(dill.dumps(obj4()))
    assert type(obj4_copy) is type(obj4_copy).__init__.__closure__[0].cell_contents
    assert type(obj4_copy.b) is type(obj4_copy.b).__init__.__closure__[0].cell_contents