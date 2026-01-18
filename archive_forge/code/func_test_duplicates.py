import unittest
from os import sys, path
def test_duplicates(self, error_msg, option, val):
    if option not in ('implies', 'headers', 'flags', 'group', 'detect', 'extra_checks'):
        return
    if isinstance(val, str):
        val = val.split()
    if len(val) != len(set(val)):
        raise AssertionError(error_msg + "duplicated values in option '%s'" % option)