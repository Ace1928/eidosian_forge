import pickle
import copy
from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .. import constants
from .common import TestCase
def test_copy_uncertainquantity(self):
    for dtype in [float, object]:
        x = UncertainQuantity(20, 'm', 0.2).astype(dtype)
        y = copy.copy(x)
        self.assertQuantityEqual(x, y)