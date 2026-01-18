import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def get_attribute_list(self, key, default=None):
    """The same as get(), but always returns a list.

        :param key: The attribute to look for.
        :param default: Use this value if the attribute is not present
            on this PageElement.
        :return: A list of values, probably containing only a single
            value.
        """
    value = self.get(key, default)
    if not isinstance(value, list):
        value = [value]
    return value