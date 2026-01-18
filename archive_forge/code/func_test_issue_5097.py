import unittest
from numba.tests.support import TestCase
from numba.core.compiler import run_frontend
def test_issue_5097(self):

    def udt():
        for i in range(0):
            if i > 0:
                pass
            a = None
    run_frontend(udt)