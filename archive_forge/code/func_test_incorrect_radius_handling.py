import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def test_incorrect_radius_handling(self):
    with assert_raises(ValueError):
        SphericalVoronoi(self.points, radius=0.98)