import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def get_shelf_filename(self, shelf_id):
    return 'shelf-%d' % shelf_id