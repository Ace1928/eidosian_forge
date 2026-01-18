import unittest
import numpy as np
from pygsp import graphs, filters
def test_meyer(self):
    f = filters.Meyer(self._G, Nf=4)
    self._test_methods(f, tight=True)