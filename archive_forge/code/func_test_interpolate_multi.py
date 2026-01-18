import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_interpolate_multi(self):
    x = [10.0, 19.0, 27.1]
    y = [-10.0, -19.0, -27.1]
    p = (0.0, 0.0)
    for i in range(0, 3):
        p = interpolate(p, [100, -100])
        self.assertEqual(p, [x[i], y[i]])