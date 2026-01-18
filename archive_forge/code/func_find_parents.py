import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def find_parents(self, name=None, attrs={}, limit=None, **kwargs):
    """Find all parents of this PageElement that match the given criteria.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: A dictionary of filters on attribute values.
        :param limit: Stop looking after finding this many results.
        :kwargs: A dictionary of filters on attribute values.

        :return: A PageElement.
        :rtype: bs4.element.Tag | bs4.element.NavigableString
        """
    _stacklevel = kwargs.pop('_stacklevel', 2)
    return self._find_all(name, attrs, None, limit, self.parents, _stacklevel=_stacklevel + 1, **kwargs)