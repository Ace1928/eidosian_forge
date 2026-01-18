import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testEnumName(self):
    """Test enum name."""
    module_name = test_util.get_module_name(EnumTest)
    self.assertEquals('%s.Color' % module_name, Color.definition_name())
    self.assertEquals(module_name, Color.outer_definition_name())
    self.assertEquals(module_name, Color.definition_package())