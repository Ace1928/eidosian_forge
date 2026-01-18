import difflib
import math
from ..compat import collections_abc
import six
from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import text_format
def checkFloatEqAndReplace(self, expected, actual, relative_tolerance):
    """Recursively replaces the floats in actual with those in expected iff they are approximately equal.

  This is done because string equality will consider values such as 5.0999999999
  and 5.1 as not being equal, despite being extremely close.

  Args:
    self: googletest.TestCase
    expected: expected values
    actual: actual values
    relative_tolerance: float, relative tolerance.
  """
    for expected_fields, actual_fields in zip(expected.ListFields(), actual.ListFields()):
        is_repeated = True
        expected_desc, expected_values = expected_fields
        actual_values = actual_fields[1]
        if expected_desc.label != descriptor.FieldDescriptor.LABEL_REPEATED:
            is_repeated = False
            expected_values = [expected_values]
            actual_values = [actual_values]
        if expected_desc.type == descriptor.FieldDescriptor.TYPE_FLOAT or expected_desc.type == descriptor.FieldDescriptor.TYPE_DOUBLE:
            for i, (x, y) in enumerate(zip(expected_values, actual_values)):
                if isClose(x, y, relative_tolerance):
                    if is_repeated:
                        getattr(actual, actual_fields[0].name)[i] = x
                    else:
                        setattr(actual, actual_fields[0].name, x)
        if expected_desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE or expected_desc.type == descriptor.FieldDescriptor.TYPE_GROUP:
            if expected_desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE and expected_desc.message_type.has_options and expected_desc.message_type.GetOptions().map_entry:
                if expected_desc.message_type.fields_by_number[2].type == descriptor.FieldDescriptor.TYPE_MESSAGE:
                    for e_v, a_v in zip(six.itervalues(expected_values), six.itervalues(actual_values)):
                        checkFloatEqAndReplace(self, expected=e_v, actual=a_v, relative_tolerance=relative_tolerance)
            else:
                for v, a in zip(expected_values, actual_values):
                    checkFloatEqAndReplace(self, expected=v, actual=a, relative_tolerance=relative_tolerance)