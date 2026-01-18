import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_get_random_color_random_alpha(self):
    actual = get_random_color(alpha='random')
    self.assertEqual(len(actual), 4)