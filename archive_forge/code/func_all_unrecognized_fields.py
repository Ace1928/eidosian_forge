import types
import weakref
import six
from apitools.base.protorpclite import util
def all_unrecognized_fields(self):
    """Get the names of all unrecognized fields in this message."""
    return list(self.__unrecognized_fields.keys())