import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def makeSuiteForClasses(*classes, testMethodPrefix=None):
    """Return a test suite with tests loaded from provided classes."""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    if testMethodPrefix:
        loader.testMethodPrefix = testMethodPrefix
    for C in classes:
        suite.addTest(loader.loadTestsFromTestCase(C))
    return suite