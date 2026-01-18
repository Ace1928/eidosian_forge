from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def make_tests():
    leading_sign = ['+', '-', '']
    leading_digit = ['0', '']
    dot = ['.', '']
    decimal_digit = ['1', '']
    e = ['e', 'E', '']
    e_sign = ['+', '-', '']
    e_int = ['22', '']
    stray = ['9', '.', '']
    seen = set()
    seen.add('')
    for parts in product(leading_sign, stray, leading_digit, dot, decimal_digit, stray, e, e_sign, e_int, stray):
        parts_str = ''.join(parts).strip()
        if parts_str in seen:
            continue
        seen.add(parts_str)
        yield parts_str
    print_(len(seen) - 1, 'tests produced')