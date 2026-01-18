from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def bump_num(self, matchobj):
    int_value = int(matchobj[0])
    return str(int_value + 1)