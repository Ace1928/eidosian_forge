import unittest
import numpy as np
from pygsp import graphs, filters
def test_expwin(self):
    f = filters.Expwin(self._G)
    self._test_methods(f, tight=False)