import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def find_next_sibling(self, name=None, attrs={}, string=None, **kwargs):
    """Find the closest sibling to this PageElement that matches the
        given criteria and appears later in the document.

        All find_* methods take a common set of arguments. See the
        online documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: A dictionary of filters on attribute values.
        :param string: A filter for a NavigableString with specific text.
        :kwargs: A dictionary of filters on attribute values.
        :return: A PageElement.
        :rtype: bs4.element.Tag | bs4.element.NavigableString
        """
    return self._find_one(self.find_next_siblings, name, attrs, string, **kwargs)