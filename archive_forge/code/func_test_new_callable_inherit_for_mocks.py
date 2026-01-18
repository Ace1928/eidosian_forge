import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_new_callable_inherit_for_mocks(self):

    class MockSub(Mock):
        pass
    MockClasses = (NonCallableMock, NonCallableMagicMock, MagicMock, Mock, MockSub)
    for Klass in MockClasses:
        for arg in ('spec', 'spec_set'):
            kwargs = {arg: True}
            p = patch(foo_name, new_callable=Klass, **kwargs)
            m = p.start()
            try:
                instance = m.return_value
                self.assertRaises(AttributeError, getattr, instance, 'x')
            finally:
                p.stop()