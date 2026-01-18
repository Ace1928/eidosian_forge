import sys
import os
import re
import warnings
import types
import unicodedata
@classmethod
def is_not_known_attribute(cls, attr):
    """
        Returns True if and only if the given attribute is NOT recognized by
        this class.
        """
    return attr not in cls.known_attributes