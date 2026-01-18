import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_getattribute(self):

    class Replaced:
        foo = 'bar'

    def factory(*args):
        return Replaced()
    replacer = lazy_import.ScopeReplacer({}, factory, 'name')

    def racer():
        replacer.foo
    self.run_race(racer)