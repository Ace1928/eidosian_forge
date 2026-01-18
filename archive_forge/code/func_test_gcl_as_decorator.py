import unittest
from numba.core.compiler_lock import (
from numba.tests.support import TestCase
def test_gcl_as_decorator(self):

    @global_compiler_lock
    def func():
        require_global_compiler_lock()
    func()