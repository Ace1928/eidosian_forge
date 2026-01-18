import types
import weakref
import six
from apitools.base.protorpclite import util
@classmethod
def all_fields(cls):
    """Get all field definition objects.

        Ordering is arbitrary.

        Returns:
          Iterator over all values in arbitrary order.
        """
    return cls.__by_name.values()