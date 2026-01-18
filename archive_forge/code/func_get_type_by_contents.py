import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
def get_type_by_contents(path, max_pri=100, min_pri=0):
    """Returns type of file by its contents, or None if not known"""
    update_cache()
    return magic.match(path, max_pri, min_pri)