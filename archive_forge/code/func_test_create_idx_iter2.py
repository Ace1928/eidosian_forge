import unittest
import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
@pytest.mark.parametrize('orientation', ['lr-tb', 'lr-bt', 'rl-tb', 'rl-bt', 'tb-lr', 'tb-rl', 'bt-lr', 'bt-rl'])
def test_create_idx_iter2(orientation):
    from kivy.uix.gridlayout import GridLayout
    gl = GridLayout(orientation=orientation)
    index_iter = gl._create_idx_iter(1, 1)
    assert [(0, 0)] == list(index_iter)