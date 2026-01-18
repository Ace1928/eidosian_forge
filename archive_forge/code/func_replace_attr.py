import sys
import os
import re
import warnings
import types
import unicodedata
def replace_attr(self, attr, value, force=True):
    """
        If self[attr] does not exist or force is True or omitted, set
        self[attr] to value, otherwise do nothing.
        """
    if force or self.get(attr) is None:
        self[attr] = value