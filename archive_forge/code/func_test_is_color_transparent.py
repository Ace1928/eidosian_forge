import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_is_color_transparent(self):
    c = [1, 1, 1]
    self.assertFalse(is_color_transparent(c))
    c = [1, 1, 1, 1]
    self.assertFalse(is_color_transparent(c))
    c = [1, 1, 1, 0]
    self.assertTrue(is_color_transparent(c))