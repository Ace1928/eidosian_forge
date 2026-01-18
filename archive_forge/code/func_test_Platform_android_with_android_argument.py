import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_Platform_android_with_android_argument(self):
    with patch.dict('os.environ', {'ANDROID_ARGUMENT': ''}):
        self.assertEqual(_get_platform(), 'android')
    self.assertNotIn('ANDROID_ARGUMENT', os.environ)