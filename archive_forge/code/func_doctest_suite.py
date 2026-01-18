import doctest
import re
import sys
def doctest_suite(module, **kwargs):
    return doctest.DocTestSuite(module, checker=Py23DocChecker(), **kwargs)