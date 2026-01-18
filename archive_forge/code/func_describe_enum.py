import codecs
import types
import six
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def describe_enum(enum_definition):
    """Build descriptor for Enum class.

    Args:
      enum_definition: Enum class to provide descriptor for.

    Returns:
      Initialized EnumDescriptor instance describing the Enum class.
    """
    enum_descriptor = EnumDescriptor()
    enum_descriptor.name = enum_definition.definition_name().split('.')[-1]
    values = []
    for number in sorted(enum_definition.numbers()):
        value = enum_definition.lookup_by_number(number)
        values.append(describe_enum_value(value))
    if values:
        enum_descriptor.values = values
    return enum_descriptor