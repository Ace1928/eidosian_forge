import unittest
import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
@pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
def test_2x2_rl_tb(self, n_cols, n_rows):
    assert [(100, 100), (0, 100), (100, 0), (0, 0)] == self.compute_layout(n_children=4, ori='rl-tb', n_cols=n_cols, n_rows=n_rows)