import unittest
import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
@pytest.mark.parametrize('n_cols, n_rows, orientation, expectation', [(2, 3, 'lr-tb', [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]), (2, 3, 'lr-bt', [(0, 2), (1, 2), (0, 1), (1, 1), (0, 0), (1, 0)]), (2, 3, 'rl-tb', [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2)]), (2, 3, 'rl-bt', [(1, 2), (0, 2), (1, 1), (0, 1), (1, 0), (0, 0)]), (2, 3, 'tb-lr', [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]), (2, 3, 'tb-rl', [(1, 0), (1, 1), (1, 2), (0, 0), (0, 1), (0, 2)]), (2, 3, 'bt-lr', [(0, 2), (0, 1), (0, 0), (1, 2), (1, 1), (1, 0)]), (2, 3, 'bt-rl', [(1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)])])
def test_create_idx_iter(n_cols, n_rows, orientation, expectation):
    from kivy.uix.gridlayout import GridLayout
    gl = GridLayout(orientation=orientation)
    index_iter = gl._create_idx_iter(n_cols, n_rows)
    assert expectation == list(index_iter)