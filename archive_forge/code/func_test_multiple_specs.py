import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_multiple_specs(self):
    original = PTModule
    for kwarg in ('spec', 'spec_set'):
        p = patch(MODNAME, autospec=0, **{kwarg: 0})
        self.assertRaises(TypeError, p.start)
        self.assertIs(PTModule, original)
    for kwarg in ('spec', 'autospec'):
        p = patch(MODNAME, spec_set=0, **{kwarg: 0})
        self.assertRaises(TypeError, p.start)
        self.assertIs(PTModule, original)
    for kwarg in ('spec_set', 'autospec'):
        p = patch(MODNAME, spec=0, **{kwarg: 0})
        self.assertRaises(TypeError, p.start)
        self.assertIs(PTModule, original)