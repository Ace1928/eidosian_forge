from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def testMatch(expression, instring, shouldPass, expectedString=None):
    if shouldPass:
        try:
            result = expression.parseString(instring)
            print_('%s correctly matched %s' % (repr(expression), repr(instring)))
            if expectedString != result[0]:
                print_('\tbut failed to match the pattern as expected:')
                print_('\tproduced %s instead of %s' % (repr(result[0]), repr(expectedString)))
            return True
        except pp.ParseException:
            print_('%s incorrectly failed to match %s' % (repr(expression), repr(instring)))
    else:
        try:
            result = expression.parseString(instring)
            print_('%s incorrectly matched %s' % (repr(expression), repr(instring)))
            print_('\tproduced %s as a result' % repr(result[0]))
        except pp.ParseException:
            print_('%s correctly failed to match %s' % (repr(expression), repr(instring)))
            return True
    return False