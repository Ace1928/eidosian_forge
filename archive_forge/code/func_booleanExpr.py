from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def booleanExpr(atom):
    ops = [(supLiteral('!'), 1, pp.opAssoc.RIGHT, lambda s, l, t: ['!', t[0][0]]), (pp.oneOf('= !='), 2, pp.opAssoc.LEFT), (supLiteral('&'), 2, pp.opAssoc.LEFT, lambda s, l, t: ['&', t[0]]), (supLiteral('|'), 2, pp.opAssoc.LEFT, lambda s, l, t: ['|', t[0]])]
    return pp.infixNotation(atom, ops)