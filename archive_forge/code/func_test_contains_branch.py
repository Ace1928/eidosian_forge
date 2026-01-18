import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_contains_branch(self):
    r1 = self.ta2 + self.ta1
    r2 = self.ta2 + self.ta1
    assert r1 == r2
    assert r1 != self.ta1
    assert r1.contains_branch(r2)
    assert r1.contains_branch(self.ta1)
    assert not r1.contains_branch(self.ta2)
    assert not r1.contains_branch(self.ta2 + self.ta2)
    assert r1 == r2
    assert self.stack1.contains_branch(self.ta3)
    assert self.stack2.contains_branch(self.ta3)
    assert self.stack1.contains_branch(self.stack2_subset)
    assert self.stack2.contains_branch(self.stack2_subset)
    assert not self.stack2_subset.contains_branch(self.stack1)
    assert not self.stack2_subset.contains_branch(self.stack2)
    assert self.stack1.contains_branch(self.ta2 + self.ta3)
    assert self.stack2.contains_branch(self.ta2 + self.ta3)
    assert not self.stack1.contains_branch(self.tn1 + self.ta2)