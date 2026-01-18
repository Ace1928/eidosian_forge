import functools
import sys
import types
import warnings
import unittest
def reversed_cmp(x, y):
    return -((x > y) - (x < y))