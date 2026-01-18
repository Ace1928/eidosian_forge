import sys
import os
import re
import warnings
import types
import unicodedata
def append_attr_list(self, attr, values):
    """
        For each element in values, if it does not exist in self[attr], append
        it.

        NOTE: Requires self[attr] and values to be sequence type and the
        former should specifically be a list.
        """
    for value in values:
        if not value in self[attr]:
            self[attr].append(value)