from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def runTest2(self):
    from pyparsing import Word, alphanums, OneOrMore, Group, Regex, Optional
    word = Word(alphanums + '_').setName('word')
    with_stmt = 'with' + OneOrMore(Group(word('key') + '=' + word('value')))('overrides')
    using_stmt = 'using' + Regex('id-[0-9a-f]{8}')('id')
    modifiers = Optional(with_stmt('with_stmt')) & Optional(using_stmt('using_stmt'))
    self.assertEqual(modifiers, 'with foo=bar bing=baz using id-deadbeef')
    self.assertNotEqual(modifiers, 'with foo=bar bing=baz using id-deadbeef using id-feedfeed')