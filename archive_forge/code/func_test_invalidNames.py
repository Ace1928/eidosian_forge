import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_invalidNames(self):
    """
        Passing a name which isn't a fully-qualified Python name to L{namedAny}
        should result in one of the following exceptions:
         - L{InvalidName}: the name is not a dot-separated list of Python
           objects
         - L{ObjectNotFound}: the object doesn't exist
         - L{ModuleNotFound}: the object doesn't exist and there is only one
           component in the name
        """
    err = self.assertRaises(reflect.ModuleNotFound, reflect.namedAny, 'nosuchmoduleintheworld')
    self.assertEqual(str(err), "No module named 'nosuchmoduleintheworld'")
    err = self.assertRaises(reflect.ObjectNotFound, reflect.namedAny, '@#$@(#.!@(#!@#')
    self.assertEqual(str(err), "'@#$@(#.!@(#!@#' does not name an object")
    err = self.assertRaises(reflect.ObjectNotFound, reflect.namedAny, 'tcelfer.nohtyp.detsiwt')
    self.assertEqual(str(err), "'tcelfer.nohtyp.detsiwt' does not name an object")
    err = self.assertRaises(reflect.InvalidName, reflect.namedAny, '')
    self.assertEqual(str(err), 'Empty module name')
    for invalidName in ['.twisted', 'twisted.', 'twisted..python']:
        err = self.assertRaises(reflect.InvalidName, reflect.namedAny, invalidName)
        self.assertEqual(str(err), "name must be a string giving a '.'-separated list of Python identifiers, not %r" % (invalidName,))