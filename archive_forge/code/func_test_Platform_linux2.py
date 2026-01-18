import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_Platform_linux2(self):
    self._test_platforms('linux2', 'linux')