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
def test_left_to_right_iteration(self):
    stack3 = self.ta1 + (self.tn1 + (self.ta2 + self.tn2)) + self.ta3
    target_transforms = [stack3, self.tn1 + (self.ta2 + self.tn2) + self.ta3, self.ta2 + self.tn2 + self.ta3, self.tn2 + self.ta3, self.ta3]
    r = [rh for _, rh in stack3._iter_break_from_left_to_right()]
    assert len(r) == len(target_transforms)
    for target_stack, stack in zip(target_transforms, r):
        assert target_stack == stack