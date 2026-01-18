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
def testUnknownValues(self):
    """Test message class equality with unknown fields."""

    class MyMessage(messages.Message):
        field1 = messages.IntegerField(1)
    message = MyMessage()
    self.assertEquals([], message.all_unrecognized_fields())
    self.assertEquals((None, None), message.get_unrecognized_field_info('doesntexist'))
    self.assertEquals((None, None), message.get_unrecognized_field_info('doesntexist', None, None))
    self.assertEquals(('defaultvalue', 'defaultwire'), message.get_unrecognized_field_info('doesntexist', 'defaultvalue', 'defaultwire'))
    self.assertEquals((3, None), message.get_unrecognized_field_info('doesntexist', value_default=3))
    message.set_unrecognized_field('exists', 9.5, messages.Variant.DOUBLE)
    self.assertEquals(1, len(message.all_unrecognized_fields()))
    self.assertTrue('exists' in message.all_unrecognized_fields())
    self.assertEquals((9.5, messages.Variant.DOUBLE), message.get_unrecognized_field_info('exists'))
    self.assertEquals((9.5, messages.Variant.DOUBLE), message.get_unrecognized_field_info('exists', 'type', 1234))
    self.assertEquals((1234, None), message.get_unrecognized_field_info('doesntexist', 1234))
    message.set_unrecognized_field('another', 'value', messages.Variant.STRING)
    self.assertEquals(2, len(message.all_unrecognized_fields()))
    self.assertTrue('exists' in message.all_unrecognized_fields())
    self.assertTrue('another' in message.all_unrecognized_fields())
    self.assertEquals((9.5, messages.Variant.DOUBLE), message.get_unrecognized_field_info('exists'))
    self.assertEquals(('value', messages.Variant.STRING), message.get_unrecognized_field_info('another'))
    message.set_unrecognized_field('typetest1', ['list', 0, ('test',)], messages.Variant.STRING)
    self.assertEquals((['list', 0, ('test',)], messages.Variant.STRING), message.get_unrecognized_field_info('typetest1'))
    message.set_unrecognized_field('typetest2', '', messages.Variant.STRING)
    self.assertEquals(('', messages.Variant.STRING), message.get_unrecognized_field_info('typetest2'))