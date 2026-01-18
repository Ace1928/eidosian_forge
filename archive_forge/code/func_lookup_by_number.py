import types
import weakref
import six
from apitools.base.protorpclite import util
def lookup_by_number(cls, number):
    """Look up Enum by number.

        Args:
          number: Number of enum to find.

        Returns:
          Enum sub-class instance of that value.
        """
    return cls.__by_number[number]