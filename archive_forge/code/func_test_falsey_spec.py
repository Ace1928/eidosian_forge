import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_falsey_spec(self):
    for kwarg in ('spec', 'autospec', 'spec_set'):
        p = patch(MODNAME, **{kwarg: 0})
        m = p.start()
        try:
            self.assertRaises(AttributeError, getattr, m, 'doesnotexit')
        finally:
            p.stop()