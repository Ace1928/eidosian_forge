import types
import weakref
import six
from apitools.base.protorpclite import util
@classmethod
def field_by_name(cls, name):
    """Get field by name.

        Returns:
          Field object associated with name.

        Raises:
          KeyError if no field found by that name.
        """
    return cls.__by_name[name]