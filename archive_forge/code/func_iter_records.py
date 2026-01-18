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
def iter_records(self):
    """Iterate through bundle records

        :return: a generator of (bytes, metadata, content_kind, revision_id,
            file_id)
        """
    iterator = pack.iter_records_from_file(self._container_file)
    for names, bytes in iterator:
        if len(names) != 1:
            raise errors.BadBundle('Record has %d names instead of 1' % len(names))
        metadata = bencode.bdecode(bytes)
        if metadata[b'storage_kind'] == b'header':
            bytes = None
        else:
            _unused, bytes = next(iterator)
        yield ((bytes, metadata) + self.decode_name(names[0][0]))