import sys
import os
import re
import warnings
import types
import unicodedata
def coerce_append_attr_list(self, attr, value):
    """
        First, convert both self[attr] and value to a non-string sequence
        type; if either is not already a sequence, convert it to a list of one
        element.  Then call append_attr_list.

        NOTE: self[attr] and value both must not be None.
        """
    if not isinstance(self.get(attr), list):
        self[attr] = [self[attr]]
    if not isinstance(value, list):
        value = [value]
    self.append_attr_list(attr, value)