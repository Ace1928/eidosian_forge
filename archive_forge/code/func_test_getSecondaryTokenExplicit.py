from yowsup.layers.coder.tokendictionary import TokenDictionary
import unittest
def test_getSecondaryTokenExplicit(self):
    self.assertEqual(self.tokenDictionary.getToken(11, True), 'reject')