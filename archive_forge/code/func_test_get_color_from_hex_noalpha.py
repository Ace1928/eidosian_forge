import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_get_color_from_hex_noalpha(self):
    actual = get_color_from_hex('#d1a9c4')
    expected = [0.81960784, 0.66274509, 0.76862745, 1.0]
    for i in range(4):
        self.assertAlmostEqual(actual[i], expected[i])