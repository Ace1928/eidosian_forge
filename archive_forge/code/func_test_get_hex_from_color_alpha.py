import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_get_hex_from_color_alpha(self):
    actual = get_hex_from_color([0.25, 0.77, 0.9, 0.5])
    expected = '#3fc4e57f'
    self.assertEqual(actual, expected)