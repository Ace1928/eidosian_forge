import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testLabel(self):
    for repeated, required, expected_label in ((True, False, descriptor.FieldDescriptor.Label.REPEATED), (False, True, descriptor.FieldDescriptor.Label.REQUIRED), (False, False, descriptor.FieldDescriptor.Label.OPTIONAL)):
        field = messages.IntegerField(10, required=required, repeated=repeated)
        field.name = 'a_field'
        expected = descriptor.FieldDescriptor()
        expected.name = 'a_field'
        expected.number = 10
        expected.label = expected_label
        expected.variant = descriptor.FieldDescriptor.Variant.INT64
        described = descriptor.describe_field(field)
        described.check_initialized()
        self.assertEquals(expected, described)