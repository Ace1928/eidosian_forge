import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_enable_proxying(self):
    """Test that we can allow ScopeReplacer to proxy."""
    actions = []
    InstrumentedReplacer.use_actions(actions)
    TestClass.use_actions(actions)

    def factory(replacer, scope, name):
        actions.append('factory')
        return TestClass()
    try:
        test_obj4
    except NameError:
        pass
    else:
        self.fail('test_obj4 was not supposed to exist yet')
    lazy_import.ScopeReplacer._should_proxy = True
    InstrumentedReplacer(scope=globals(), name='test_obj4', factory=factory)
    self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj4, '__class__'))
    test_obj5 = test_obj4
    self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj4, '__class__'))
    self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj5, '__class__'))
    self.assertEqual('foo', test_obj4.foo(1))
    self.assertEqual(TestClass, object.__getattribute__(test_obj4, '__class__'))
    self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj5, '__class__'))
    self.assertEqual('foo', test_obj4.foo(2))
    self.assertEqual('foo', test_obj5.foo(3))
    self.assertEqual('foo', test_obj5.foo(4))
    self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj5, '__class__'))
    self.assertEqual([('__getattribute__', 'foo'), 'factory', 'init', ('foo', 1), ('foo', 2), ('__getattribute__', 'foo'), ('foo', 3), ('__getattribute__', 'foo'), ('foo', 4)], actions)