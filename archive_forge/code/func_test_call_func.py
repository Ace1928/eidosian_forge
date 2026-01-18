import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_call_func(self):
    actions = []
    InstrumentedReplacer.use_actions(actions)

    def func(a, b, c=None):
        actions.append('func')
        return (a, b, c)

    def factory(replacer, scope, name):
        actions.append('factory')
        return func
    try:
        test_func1
    except NameError:
        pass
    else:
        self.fail('test_func1 was not supposed to exist yet')
    InstrumentedReplacer(scope=globals(), name='test_func1', factory=factory)
    self.assertFalse(test_func1 is func)
    val = test_func1(1, 2, c='3')
    self.assertIs(test_func1, func)
    self.assertEqual((1, 2, '3'), val)
    self.assertEqual([('__call__', (1, 2), {'c': '3'}), 'factory', 'func'], actions)