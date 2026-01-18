import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def test_angle_axis():
    for M, q in eg_pairs:
        theta, vec = nq.quat2angle_axis(q)
        q2 = nq.angle_axis2quat(theta, vec)
        nq.nearly_equivalent(q, q2)
        aa_mat = nq.angle_axis2mat(theta, vec)
        assert_array_almost_equal(aa_mat, M)
        unit_vec = norm(vec)
        aa_mat2 = nq.angle_axis2mat(theta, unit_vec, is_normalized=True)
        assert_array_almost_equal(aa_mat2, M)