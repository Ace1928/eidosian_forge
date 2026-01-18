import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
def first_match(self, path):
    """Return the first match found for a given path, or None if no match
        is found."""
    try:
        return next(self._match_path(path))[0]
    except StopIteration:
        return None