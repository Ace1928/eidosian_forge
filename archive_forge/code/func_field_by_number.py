import types
import weakref
import six
from apitools.base.protorpclite import util
@classmethod
def field_by_number(cls, number):
    """Get field by number.

        Returns:
          Field object associated with number.

        Raises:
          KeyError if no field found by that number.
        """
    return cls.__by_number[number]