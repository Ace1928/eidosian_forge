import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
def getTestCaseNames(self, test_case_class):
    test_fn_names = self.test_func_names.get(test_case_class, None)
    if test_fn_names is not None:
        return test_fn_names
    test_fn_names = unittest.TestLoader.getTestCaseNames(self, test_case_class)
    self.test_func_names[test_case_class] = test_fn_names
    return test_fn_names