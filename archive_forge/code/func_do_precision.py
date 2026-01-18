import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def do_precision(self, float_small, float_large):
    self.do_precision_lower_bound(float_small, float_large)
    self.do_precision_upper_bound(float_small, float_large)