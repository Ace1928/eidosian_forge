from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def runTest4(self):
    from pyparsing import pyparsing_common, ZeroOrMore, Group
    expr = ~pyparsing_common.iso8601_date + pyparsing_common.integer('id') & ZeroOrMore(Group(pyparsing_common.iso8601_date)('date*'))
    expr.runTests('\n            1999-12-31 100 2001-01-01\n            42\n            ')