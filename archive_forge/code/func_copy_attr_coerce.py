import sys
import os
import re
import warnings
import types
import unicodedata
def copy_attr_coerce(self, attr, value, replace):
    """
        If attr is an attribute of self and either self[attr] or value is a
        list, convert all non-sequence values to a sequence of 1 element and
        then concatenate the two sequence, setting the result to self[attr].
        If both self[attr] and value are non-sequences and replace is True or
        self[attr] is None, replace self[attr] with value. Otherwise, do
        nothing.
        """
    if self.get(attr) is not value:
        if isinstance(self.get(attr), list) or isinstance(value, list):
            self.coerce_append_attr_list(attr, value)
        else:
            self.replace_attr(attr, value, replace)