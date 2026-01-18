from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def evaluate_int(t):
    global count
    value = int(t[0])
    print_('evaluate_int', value)
    count += 1
    return value