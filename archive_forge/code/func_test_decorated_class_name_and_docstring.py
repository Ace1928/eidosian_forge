import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
def test_decorated_class_name_and_docstring(self):
    self.assertEqual(SampleList.__name__, 'SampleList')
    self.assertEqual(SampleList.__doc__, 'SampleList docstring.')