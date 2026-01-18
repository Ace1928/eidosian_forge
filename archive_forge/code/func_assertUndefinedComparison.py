import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def assertUndefinedComparison(testCase, s1, s2):
    """
    A custom assertion for L{SerialNumber} values that cannot be meaningfully
    compared.

    "Note that there are some pairs of values s1 and s2 for which s1 is not
    equal to s2, but for which s1 is neither greater than, nor less than, s2.
    An attempt to use these ordering operators on such pairs of values produces
    an undefined result."

    @see: U{https://tools.ietf.org/html/rfc1982#section-3.2}

    @param testCase: The L{unittest.TestCase} on which to call assertion
        methods.
    @type testCase: L{unittest.TestCase}

    @param s1: The first value to compare.
    @type s1: L{SerialNumber}

    @param s2: The second value to compare.
    @type s2: L{SerialNumber}
    """
    testCase.assertFalse(s1 == s2)
    testCase.assertFalse(s1 <= s2)
    testCase.assertFalse(s1 < s2)
    testCase.assertFalse(s1 > s2)
    testCase.assertFalse(s1 >= s2)