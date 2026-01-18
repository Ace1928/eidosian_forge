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
def add_multiparent_record(self, mp_bytes, sha1, parents, repo_kind, revision_id, file_id):
    """Add a record for a multi-parent diff

        :mp_bytes: A multi-parent diff, as a bytestring
        :sha1: The sha1 hash of the fulltext
        :parents: a list of revision-ids of the parents
        :repo_kind: The kind of object in the repository.  May be 'file' or
            'inventory'
        :revision_id: The revision id of the mpdiff being added.
        :file_id: The file-id of the file, or None for inventories.
        """
    metadata = {b'parents': parents, b'storage_kind': b'mpdiff', b'sha1': sha1}
    self._add_record(mp_bytes, metadata, repo_kind, revision_id, file_id)