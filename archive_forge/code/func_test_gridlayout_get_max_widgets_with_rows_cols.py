import unittest
import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
def test_gridlayout_get_max_widgets_with_rows_cols(self):
    gl = GridLayout()
    gl.rows = 5
    gl.cols = 3
    expected = 15
    value = gl.get_max_widgets()
    self.assertEqual(expected, value)