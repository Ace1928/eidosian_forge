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
def set_unique_metadata(self, namespace, name, value, others=None):
    """Add metadata if metadata with this identifier does not already exist, otherwise update existing metadata."""
    if namespace in NAMESPACES:
        namespace = NAMESPACES[namespace]
    if namespace in self.metadata and name in self.metadata[namespace]:
        self.metadata[namespace][name] = [(value, others)]
    else:
        self.add_metadata(namespace, name, value, others)