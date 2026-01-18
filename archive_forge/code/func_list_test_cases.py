import unittest
import binascii
from Cryptodome.Util.py3compat import b
def list_test_cases(class_):
    """Return a list of TestCase instances given a TestCase class

    This is useful when you have defined test* methods on your TestCase class.
    """
    return unittest.TestLoader().loadTestsFromTestCase(class_)