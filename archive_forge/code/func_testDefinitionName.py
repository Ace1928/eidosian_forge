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
def testDefinitionName(self):
    """Test message name."""

    class MyMessage(messages.Message):
        pass
    module_name = test_util.get_module_name(FieldTest)
    self.assertEquals('%s.MyMessage' % module_name, MyMessage.definition_name())
    self.assertEquals(module_name, MyMessage.outer_definition_name())
    self.assertEquals(module_name, MyMessage.definition_package())
    self.assertEquals(six.text_type, type(MyMessage.definition_name()))
    self.assertEquals(six.text_type, type(MyMessage.outer_definition_name()))
    self.assertEquals(six.text_type, type(MyMessage.definition_package()))