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
def read_epub(name, options=None):
    """
    Creates new instance of EpubBook with the content defined in the input file.

    >>> book = ebooklib.read_epub('book.epub')

    :Args:
      - name: full path to the input file
      - options: extra options as dictionary (optional)

    :Returns:
      Instance of EpubBook.
    """
    reader = EpubReader(name, options)
    book = reader.load()
    reader.process()
    return book