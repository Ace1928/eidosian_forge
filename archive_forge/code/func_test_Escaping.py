import gyp.generator.xcode as xcode
import unittest
import sys
def test_Escaping(self):
    self.assertEqual(xcode.EscapeXcodeDefine('a b"c\\'), 'a\\ b\\"c\\\\')