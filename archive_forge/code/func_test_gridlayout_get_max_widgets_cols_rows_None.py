import unittest
import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
def test_gridlayout_get_max_widgets_cols_rows_None(self):
    gl = GridLayout()
    expected = None
    value = gl.get_max_widgets()
    self.assertEqual(expected, value)