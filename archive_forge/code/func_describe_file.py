import codecs
import types
import six
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def describe_file(module):
    """Build a file from a specified Python module.

    Args:
      module: Python module to describe.

    Returns:
      Initialized FileDescriptor instance describing the module.
    """
    descriptor = FileDescriptor()
    descriptor.package = util.get_package_for_module(module)
    if not descriptor.package:
        descriptor.package = None
    message_descriptors = []
    enum_descriptors = []
    for name in sorted(dir(module)):
        value = getattr(module, name)
        if isinstance(value, type):
            if issubclass(value, messages.Message):
                message_descriptors.append(describe_message(value))
            elif issubclass(value, messages.Enum):
                enum_descriptors.append(describe_enum(value))
    if message_descriptors:
        descriptor.message_types = message_descriptors
    if enum_descriptors:
        descriptor.enum_types = enum_descriptors
    return descriptor