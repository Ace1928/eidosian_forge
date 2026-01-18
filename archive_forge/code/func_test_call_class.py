import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_call_class(self):
    actions = []
    InstrumentedReplacer.use_actions(actions)
    TestClass.use_actions(actions)

    def factory(replacer, scope, name):
        actions.append('factory')
        return TestClass
    try:
        test_class2
    except NameError:
        pass
    else:
        self.fail('test_class2 was not supposed to exist yet')
    InstrumentedReplacer(scope=globals(), name='test_class2', factory=factory)
    self.assertFalse(test_class2 is TestClass)
    obj = test_class2()
    self.assertIs(test_class2, TestClass)
    self.assertIsInstance(obj, TestClass)
    self.assertEqual('class_member', obj.class_member)
    self.assertEqual([('__call__', (), {}), 'factory', 'init'], actions)