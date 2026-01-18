import unittest
import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
@pytest.mark.parametrize('n_cols, n_rows', [(1, None), (None, 3), (1, 3)])
@pytest.mark.parametrize('ori', 'tb-lr tb-rl lr-tb rl-tb'.split())
def test_1x3_tb(self, ori, n_cols, n_rows):
    assert [(0, 200), (0, 100), (0, 0)] == self.compute_layout(n_children=3, ori=ori, n_cols=n_cols, n_rows=n_rows)