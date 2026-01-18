import pytest
import numpy as np
import cirq
from cirq_pasqal import ThreeDQubit, TwoDQubit
def test_cube_3d():
    assert ThreeDQubit.cube(2, x0=1, y0=1, z0=1) == [ThreeDQubit(1, 1, 1), ThreeDQubit(2, 1, 1), ThreeDQubit(1, 2, 1), ThreeDQubit(2, 2, 1), ThreeDQubit(1, 1, 2), ThreeDQubit(2, 1, 2), ThreeDQubit(1, 2, 2), ThreeDQubit(2, 2, 2)]
    assert ThreeDQubit.cube(2) == [ThreeDQubit(0, 0, 0), ThreeDQubit(1, 0, 0), ThreeDQubit(0, 1, 0), ThreeDQubit(1, 1, 0), ThreeDQubit(0, 0, 1), ThreeDQubit(1, 0, 1), ThreeDQubit(0, 1, 1), ThreeDQubit(1, 1, 1)]