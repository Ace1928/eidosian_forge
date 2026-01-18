import unittest
import idna.compat
def testToUnicode(self):
    self.assertEqual(idna.compat.ToUnicode(b'xn--zckzah.xn--zckzah'), 'テスト.テスト')