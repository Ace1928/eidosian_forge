import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_escape_markup(self):
    escaped = escape_markup('Sun [1] & Moon [2].')
    self.assertEqual(escaped, 'Sun &bl;1&br; &amp; Moon &bl;2&br;.')