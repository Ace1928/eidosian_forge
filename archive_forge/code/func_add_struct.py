import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def add_struct(self):
    """Appends and returns a struct value as the next value in the list."""
    struct_value = self.values.add().struct_value
    struct_value.Clear()
    return struct_value