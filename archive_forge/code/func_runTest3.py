from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def runTest3(self):
    from pyparsing import Literal, Suppress, ZeroOrMore, OneOrMore
    foo = Literal('foo')
    bar = Literal('bar')
    openBrace = Suppress(Literal('{'))
    closeBrace = Suppress(Literal('}'))
    exp = openBrace + (OneOrMore(foo)('foo') & ZeroOrMore(bar)('bar')) + closeBrace
    tests = '            {foo}\n            {bar foo bar foo bar foo}\n            '.splitlines()
    for test in tests:
        test = test.strip()
        if not test:
            continue
        result = exp.parseString(test)
        print_(test, '->', result.asList())
        self.assertEqual(result.asList(), test.strip('{}').split(), 'failed to parse Each expression %r' % test)
        print_(result.dump())
    try:
        result = exp.parseString('{bar}')
        self.assertTrue(False, 'failed to raise exception when required element is missing')
    except ParseException as pe:
        pass