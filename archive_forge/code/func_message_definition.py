import types
import weakref
import six
from apitools.base.protorpclite import util
def message_definition(self):
    """Get Message definition that contains this Field definition.

        Returns:
          Containing Message definition for Field. Will return None if
          for some reason Field is defined outside of a Message class.

        """
    try:
        return self._message_definition()
    except AttributeError:
        return None