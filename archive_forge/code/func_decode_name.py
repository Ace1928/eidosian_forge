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
@staticmethod
def decode_name(name):
    """Decode a name from its container form into a semantic form

        :retval: content_kind, revision_id, file_id
        """
    segments = re.split(b'(//?)', name)
    names = [b'']
    for segment in segments:
        if segment == b'//':
            names[-1] += b'/'
        elif segment == b'/':
            names.append(b'')
        else:
            names[-1] += segment
    content_kind = names[0]
    revision_id = None
    file_id = None
    if len(names) > 1:
        revision_id = names[1]
    if len(names) > 2:
        file_id = names[2]
    return (content_kind.decode('ascii'), revision_id, file_id)