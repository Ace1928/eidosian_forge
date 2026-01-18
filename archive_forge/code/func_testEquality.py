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
def testEquality(self):
    """Test message class equality."""

    class MyEnum(messages.Enum):
        val1 = 1
        val2 = 2

    class AnotherMessage(messages.Message):
        string = messages.StringField(1)

    class MyMessage(messages.Message):
        field1 = messages.IntegerField(1)
        field2 = messages.EnumField(MyEnum, 2)
        field3 = messages.MessageField(AnotherMessage, 3)
    message1 = MyMessage()
    self.assertNotEquals('hi', message1)
    self.assertNotEquals(AnotherMessage(), message1)
    self.assertEquals(message1, message1)
    message2 = MyMessage()
    self.assertEquals(message1, message2)
    message1.field1 = 10
    self.assertNotEquals(message1, message2)
    message2.field1 = 20
    self.assertNotEquals(message1, message2)
    message2.field1 = 10
    self.assertEquals(message1, message2)
    message1.field2 = MyEnum.val1
    self.assertNotEquals(message1, message2)
    message2.field2 = MyEnum.val2
    self.assertNotEquals(message1, message2)
    message2.field2 = MyEnum.val1
    self.assertEquals(message1, message2)
    message1.field3 = AnotherMessage()
    message1.field3.string = 'value1'
    self.assertNotEquals(message1, message2)
    message2.field3 = AnotherMessage()
    message2.field3.string = 'value2'
    self.assertNotEquals(message1, message2)
    message2.field3.string = 'value1'
    self.assertEquals(message1, message2)