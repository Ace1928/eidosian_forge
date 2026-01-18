import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def iter_files_recursive(self):
    """Walk the relative paths of all files in this transport."""
    tree = self._list_tree('.', 'Infinity')
    for name, is_dir, size, is_exex in tree:
        if not is_dir:
            yield name