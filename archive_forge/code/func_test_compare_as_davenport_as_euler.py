import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_compare_as_davenport_as_euler():
    rnd = np.random.RandomState(0)
    n = 100
    angles = np.empty((n, 3))
    angles[:, 0] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    angles[:, 1] = rnd.uniform(low=0, high=np.pi, size=(n,))
    angles[:, 2] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    for order in ['extrinsic', 'intrinsic']:
        for seq_tuple in permutations('xyz'):
            seq = ''.join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
            ax = [basis_vec(i) for i in seq]
            if order == 'intrinsic':
                seq = seq.upper()
            rot = Rotation.from_euler(seq, angles)
            eul = rot.as_euler(seq)
            dav = rot.as_davenport(ax, order)
            assert_allclose(eul, dav, rtol=1e-12)
    angles[:, 1] -= np.pi / 2
    for order in ['extrinsic', 'intrinsic']:
        for seq_tuple in permutations('xyz'):
            seq = ''.join(seq_tuple)
            ax = [basis_vec(i) for i in seq]
            if order == 'intrinsic':
                seq = seq.upper()
            rot = Rotation.from_euler(seq, angles)
            eul = rot.as_euler(seq)
            dav = rot.as_davenport(ax, order)
            assert_allclose(eul, dav, rtol=1e-12)