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
def get_items_of_type(self, item_type):
    """
        Returns all items of specified type.

        >>> book.get_items_of_type(epub.ITEM_IMAGE)

        :Args:
          - item_type: Type for items we are searching for

        :Returns:
          Returns found items as tuple.
        """
    return (item for item in self.items if item.get_type() == item_type)