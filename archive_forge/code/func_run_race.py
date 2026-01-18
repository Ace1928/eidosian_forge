import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def run_race(self, racer, method_to_trace='_resolve'):
    self.overrideAttr(lazy_import.ScopeReplacer, '_should_proxy', True)
    self.racer = racer
    self.method_to_trace = method_to_trace
    sys.settrace(self.tracer)
    self.racer()
    self.assertEqual(None, sys.gettrace())