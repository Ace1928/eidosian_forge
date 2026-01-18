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
def testDefinitionName_OverrideModule(self):
    """Test message module is overriden by module package name."""

    class MyMessage(messages.Message):
        pass
    global package
    package = 'my.package'
    try:
        self.assertEquals('my.package.MyMessage', MyMessage.definition_name())
        self.assertEquals('my.package', MyMessage.outer_definition_name())
        self.assertEquals('my.package', MyMessage.definition_package())
        self.assertEquals(six.text_type, type(MyMessage.definition_name()))
        self.assertEquals(six.text_type, type(MyMessage.outer_definition_name()))
        self.assertEquals(six.text_type, type(MyMessage.definition_package()))
    finally:
        del package