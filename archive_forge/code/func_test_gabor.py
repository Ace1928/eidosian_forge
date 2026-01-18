import unittest
import numpy as np
from pygsp import graphs, filters
def test_gabor(self):
    f = filters.Gabor(self._G, lambda x: x / (1.0 + x))
    self._test_methods(f, tight=False)