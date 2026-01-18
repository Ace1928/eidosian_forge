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
def get_items_of_media_type(self, media_type):
    """
        Returns all items of specified media type.

        :Args:
          - media_type: Media type for items we are searching for

        :Returns:
          Returns found items as tuple.
        """
    return (item for item in self.items if item.media_type == media_type)