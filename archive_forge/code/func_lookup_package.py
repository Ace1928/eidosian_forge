import codecs
import types
import six
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def lookup_package(self, definition_name):
    """Determines the package name for any definition.

        Determine the package that any definition name belongs to. May
        check parent for package name and will resolve missing
        descriptors if provided descriptor loader.

        Args:
          definition_name: Definition name to find package for.

        """
    while True:
        descriptor = self.lookup_descriptor(definition_name)
        if isinstance(descriptor, FileDescriptor):
            return descriptor.package
        else:
            index = definition_name.rfind('.')
            if index < 0:
                return None
            definition_name = definition_name[:index]