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
def testPickle(self):
    """Testing pickling and unpickling of Message instances."""
    global MyEnum
    global AnotherMessage
    global MyMessage

    class MyEnum(messages.Enum):
        val1 = 1
        val2 = 2

    class AnotherMessage(messages.Message):
        string = messages.StringField(1, repeated=True)

    class MyMessage(messages.Message):
        field1 = messages.IntegerField(1)
        field2 = messages.EnumField(MyEnum, 2)
        field3 = messages.MessageField(AnotherMessage, 3)
    message = MyMessage(field1=1, field2=MyEnum.val2, field3=AnotherMessage(string=['a', 'b', 'c']))
    message.set_unrecognized_field('exists', 'value', messages.Variant.STRING)
    message.set_unrecognized_field('repeated', ['list', 0, ('test',)], messages.Variant.STRING)
    unpickled = pickle.loads(pickle.dumps(message))
    self.assertEquals(message, unpickled)
    self.assertTrue(AnotherMessage.string is unpickled.field3.string.field)
    self.assertTrue('exists' in message.all_unrecognized_fields())
    self.assertEquals(('value', messages.Variant.STRING), message.get_unrecognized_field_info('exists'))
    self.assertEquals((['list', 0, ('test',)], messages.Variant.STRING), message.get_unrecognized_field_info('repeated'))