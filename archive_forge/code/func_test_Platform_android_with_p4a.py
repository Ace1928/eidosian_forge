import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_Platform_android_with_p4a(self):
    with patch.dict('os.environ', {'P4A_BOOTSTRAP': 'sdl2'}):
        self.assertEqual(_get_platform(), 'android')
    self.assertNotIn('P4A_BOOTSTRAP', os.environ)