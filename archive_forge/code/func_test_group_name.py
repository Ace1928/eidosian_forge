import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.constants import golden as phi
from scipy.spatial import cKDTree
@pytest.mark.parametrize('name', ['Q', ' ', 'CA', 'C ', 'DA', 'D ', 'I2', ''])
def test_group_name(name):
    with pytest.raises(ValueError, match="must be one of 'I', 'O', 'T', 'Dn', 'Cn'"):
        Rotation.create_group(name)