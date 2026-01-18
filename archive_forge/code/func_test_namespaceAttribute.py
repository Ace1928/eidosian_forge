from typing import List, Optional, Type, cast
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._logger import Logger
def test_namespaceAttribute(self) -> None:
    """
        Default namespace for classes using L{Logger} as a descriptor is the
        class name they were retrieved from.
        """
    obj = LogComposedObject()
    expectedNamespace = '{}.{}'.format(obj.__module__, obj.__class__.__name__)
    self.assertEqual(cast(TestLogger, obj.log).namespace, expectedNamespace)
    self.assertEqual(cast(Type[TestLogger], LogComposedObject.log).namespace, expectedNamespace)
    self.assertIs(cast(Type[TestLogger], LogComposedObject.log).source, LogComposedObject)
    self.assertIs(cast(TestLogger, obj.log).source, obj)
    self.assertIsNone(Logger().source)