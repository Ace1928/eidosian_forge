import types
import weakref
import six
from apitools.base.protorpclite import util
def numbers(cls):
    """Get all numbers for Enum.

        Returns:
          An iterator for all numbers of the enumeration in arbitrary order.
        """
    return cls.__by_number.keys()