import codecs
import types
import six
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def describe_enum_value(enum_value):
    """Build descriptor for Enum instance.

    Args:
      enum_value: Enum value to provide descriptor for.

    Returns:
      Initialized EnumValueDescriptor instance describing the Enum instance.
    """
    enum_value_descriptor = EnumValueDescriptor()
    enum_value_descriptor.name = six.text_type(enum_value.name)
    enum_value_descriptor.number = enum_value.number
    return enum_value_descriptor