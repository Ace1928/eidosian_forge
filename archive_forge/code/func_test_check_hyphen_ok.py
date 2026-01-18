import unittest
import idna
def test_check_hyphen_ok(self):
    self.assertTrue(idna.check_hyphen_ok('abc'))
    self.assertTrue(idna.check_hyphen_ok('a--b'))
    self.assertRaises(idna.IDNAError, idna.check_hyphen_ok, 'aa--')
    self.assertRaises(idna.IDNAError, idna.check_hyphen_ok, 'a-')
    self.assertRaises(idna.IDNAError, idna.check_hyphen_ok, '-a')