import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def get_shelf_ids(self, filenames):
    matcher = re.compile('shelf-([1-9][0-9]*)')
    shelf_ids = []
    for filename in filenames:
        match = matcher.match(filename)
        if match is not None:
            shelf_ids.append(int(match.group(1)))
    return shelf_ids