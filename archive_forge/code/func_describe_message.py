import codecs
import types
import six
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def describe_message(message_definition):
    """Build descriptor for Message class.

    Args:
      message_definition: Message class to provide descriptor for.

    Returns:
      Initialized MessageDescriptor instance describing the Message class.
    """
    message_descriptor = MessageDescriptor()
    message_descriptor.name = message_definition.definition_name().split('.')[-1]
    fields = sorted(message_definition.all_fields(), key=lambda v: v.number)
    if fields:
        message_descriptor.fields = [describe_field(field) for field in fields]
    try:
        nested_messages = message_definition.__messages__
    except AttributeError:
        pass
    else:
        message_descriptors = []
        for name in nested_messages:
            value = getattr(message_definition, name)
            message_descriptors.append(describe_message(value))
        message_descriptor.message_types = message_descriptors
    try:
        nested_enums = message_definition.__enums__
    except AttributeError:
        pass
    else:
        enum_descriptors = []
        for name in nested_enums:
            value = getattr(message_definition, name)
            enum_descriptors.append(describe_enum(value))
        message_descriptor.enum_types = enum_descriptors
    return message_descriptor