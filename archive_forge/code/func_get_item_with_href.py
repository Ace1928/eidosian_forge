import zipfile
import six
import logging
import uuid
import warnings
import posixpath as zip_path
import os.path
from collections import OrderedDict
from lxml import etree
import ebooklib
from ebooklib.utils import parse_string, parse_html_string, guess_type, get_pages_for_items
def get_item_with_href(self, href):
    """
        Returns item for defined HREF.

        >>> book.get_item_with_href('EPUB/document.xhtml')

        :Args:
          - href: HREF for the item we are searching for

        :Returns:
          Returns item object. Returns None if nothing was found.
        """
    for item in self.get_items():
        if item.get_name() == href:
            return item
    return None