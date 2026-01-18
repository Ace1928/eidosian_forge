import sys
import os
import re
import warnings
import types
import unicodedata
def copy_attr_convert(self, attr, value, replace=True):
    """
        If attr is an attribute of self, set self[attr] to
        [self[attr], value], otherwise set self[attr] to value.

        NOTE: replace is not used by this function and is kept only for
              compatibility with the other copy functions.
        """
    if self.get(attr) is not value:
        self.coerce_append_attr_list(attr, value)