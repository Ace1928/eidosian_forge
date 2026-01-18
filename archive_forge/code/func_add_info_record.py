import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
def add_info_record(self, kwargs):
    """Add an info record to the bundle

        Any parameters may be supplied, except 'self' and 'storage_kind'.
        Values must be lists, strings, integers, dicts, or a combination.
        """
    kwargs[b'storage_kind'] = b'header'
    self._add_record(None, kwargs, 'info', None, None)