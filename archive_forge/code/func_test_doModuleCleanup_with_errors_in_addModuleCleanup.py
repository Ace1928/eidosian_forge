import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_doModuleCleanup_with_errors_in_addModuleCleanup(self):
    module_cleanups = []

    def module_cleanup_good(*args, **kwargs):
        module_cleanups.append((3, args, kwargs))

    def module_cleanup_bad(*args, **kwargs):
        raise CustomError('CleanUpExc')

    class Module(object):
        unittest.addModuleCleanup(module_cleanup_good, 1, 2, 3, four='hello', five='goodbye')
        unittest.addModuleCleanup(module_cleanup_bad)
    self.assertEqual(unittest.case._module_cleanups, [(module_cleanup_good, (1, 2, 3), dict(four='hello', five='goodbye')), (module_cleanup_bad, (), {})])
    with self.assertRaises(CustomError) as e:
        unittest.case.doModuleCleanups()
    self.assertEqual(str(e.exception), 'CleanUpExc')
    self.assertEqual(unittest.case._module_cleanups, [])