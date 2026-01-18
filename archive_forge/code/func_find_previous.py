import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def find_previous(self, name=None, attrs={}, string=None, **kwargs):
    """Look backwards in the document from this PageElement and find the
        first PageElement that matches the given criteria.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: A dictionary of filters on attribute values.
        :param string: A filter for a NavigableString with specific text.
        :kwargs: A dictionary of filters on attribute values.
        :return: A PageElement.
        :rtype: bs4.element.Tag | bs4.element.NavigableString
        """
    return self._find_one(self.find_all_previous, name, attrs, string, **kwargs)