import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_boundary(self):
    x = boundary(-1000, 0, 100)
    self.assertEqual(x, 0)
    x = boundary(1000, 0, 100)
    self.assertEqual(x, 100)
    x = boundary(50, 0, 100)
    self.assertEqual(x, 50)