import os
import tempfile
import unittest
from Cython.Shadow import inline
from Cython.Build.Inline import safe_type
from Cython.TestUtils import CythonTest
def test_lang_version(self):
    inline_divcode = 'def f(int a, int b): return a/b'
    self.assertEqual(inline(inline_divcode, language_level=2)['f'](5, 2), 2)
    self.assertEqual(inline(inline_divcode, language_level=3)['f'](5, 2), 2.5)
    self.assertEqual(inline(inline_divcode, language_level=2)['f'](5, 2), 2)