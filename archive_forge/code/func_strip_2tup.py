import unittest
from Cython.Compiler import Code, UtilityCode
def strip_2tup(tup):
    return (tup[0] and tup[0].strip(), tup[1] and tup[1].strip())