import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
def get_type_by_data(data, max_pri=100, min_pri=0):
    """Returns type of the data, which should be bytes."""
    update_cache()
    return magic.match_data(data, max_pri, min_pri)