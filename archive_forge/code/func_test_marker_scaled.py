import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
import pytest
def test_marker_scaled():
    marker = markers.MarkerStyle('1')
    new_marker = marker.scaled(2)
    assert new_marker is not marker
    assert new_marker.get_user_transform() == Affine2D().scale(2)
    assert marker._user_transform is not new_marker._user_transform
    new_marker = marker.scaled(2, 3)
    assert new_marker is not marker
    assert new_marker.get_user_transform() == Affine2D().scale(2, 3)
    assert marker._user_transform is not new_marker._user_transform
    marker = markers.MarkerStyle('1', transform=Affine2D().translate(1, 1))
    new_marker = marker.scaled(2)
    assert new_marker is not marker
    expected = Affine2D().translate(1, 1).scale(2)
    assert new_marker.get_user_transform() == expected
    assert marker._user_transform is not new_marker._user_transform