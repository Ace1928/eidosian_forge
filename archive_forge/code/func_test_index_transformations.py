import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_index_transformations(self):
    self.optimize_compare('ea,fb,gc,hd,abcd->efgh')
    self.optimize_compare('ea,fb,abcd,gc,hd->efgh')
    self.optimize_compare('abcd,ea,fb,gc,hd->efgh')