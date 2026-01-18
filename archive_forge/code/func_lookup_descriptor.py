import codecs
import types
import six
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def lookup_descriptor(self, definition_name):
    """Lookup descriptor by name.

        Get descriptor from library by name.  If descriptor is not found will
        attempt to find via descriptor loader if provided.

        Args:
          definition_name: Definition name to find.

        Returns:
          Descriptor that describes definition name.

        Raises:
          DefinitionNotFoundError if not descriptor exists for definition name.
        """
    try:
        return self.__descriptors[definition_name]
    except KeyError:
        pass
    if self.__descriptor_loader:
        definition = self.__descriptor_loader(definition_name)
        self.__descriptors[definition_name] = definition
        return definition
    else:
        raise messages.DefinitionNotFoundError('Could not find definition for %s' % definition_name)