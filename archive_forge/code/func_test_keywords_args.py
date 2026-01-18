import unittest2 as unittest
import doctest
import sys
import funcsigs as inspect
def test_keywords_args(self):

    def test(**kwargs):
        pass
    self.assertEqual(self.signature(test), ((('kwargs', Ellipsis, Ellipsis, 'var_keyword'),), Ellipsis))