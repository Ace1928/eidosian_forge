from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
def read_dict(self, dictionary, source='<dict>'):
    """Read configuration from a dictionary.

        Keys are section names, values are dictionaries with keys and values
        that should be present in the section. If the used dictionary type
        preserves order, sections and their keys will be added in order.

        All types held in the dictionary are converted to strings during
        reading, including section names, option names and keys.

        Optional second argument is the `source` specifying the name of the
        dictionary being read.
        """
    elements_added = set()
    for section, keys in dictionary.items():
        section = str(section)
        try:
            self.add_section(section)
        except (DuplicateSectionError, ValueError):
            if self._strict and section in elements_added:
                raise
        elements_added.add(section)
        for key, value in keys.items():
            key = self.optionxform(str(key))
            if value is not None:
                value = str(value)
            if self._strict and (section, key) in elements_added:
                raise DuplicateOptionError(section, key, source)
            elements_added.add((section, key))
            self.set(section, key, value)