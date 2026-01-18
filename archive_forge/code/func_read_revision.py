from io import BytesIO
import fastbencode as bencode
from .. import lazy_import
from breezy.bzr import (
from .. import cache_utf8, errors
from .. import revision as _mod_revision
from . import serializer
def read_revision(self, f):
    return self.read_revision_from_string(f.read())