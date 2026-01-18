import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_format_bytes_to_human(self):
    a = format_bytes_to_human(6463)
    self.assertEqual(a, '6.31 KB')
    b = format_bytes_to_human(6463, precision=4)
    self.assertEqual(b, '6.3115 KB')
    c = format_bytes_to_human(646368746541)
    self.assertEqual(c, '601.98 GB')