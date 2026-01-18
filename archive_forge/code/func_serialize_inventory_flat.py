import re
from typing import Dict, Union
from xml.etree.ElementTree import (Element, ElementTree, ParseError,
from .. import errors, lazy_regex
from . import inventory, serializer
def serialize_inventory_flat(inv, append, root_id, supported_kinds, working):
    """Serialize an inventory to a flat XML file.

    :param inv: Inventory to serialize
    :param append: Function for writing a line of output
    :param working: If True skip history data - text_sha1, text_size,
        reference_revision, symlink_target.    self._check_revisions(inv)
    """
    entries = inv.iter_entries()
    root_path, root_ie = next(entries)
    for path, ie in entries:
        if ie.parent_id != root_id:
            parent_str = b''.join([b' parent_id="', encode_and_escape(ie.parent_id), b'"'])
        else:
            parent_str = b''
        if ie.kind == 'file':
            if ie.executable:
                executable = b' executable="yes"'
            else:
                executable = b''
            if not working:
                append(b'<file%s file_id="%s" name="%s"%s revision="%s" text_sha1="%s" text_size="%d" />\n' % (executable, encode_and_escape(ie.file_id), encode_and_escape(ie.name), parent_str, encode_and_escape(ie.revision), ie.text_sha1, ie.text_size))
            else:
                append(b'<file%s file_id="%s" name="%s"%s />\n' % (executable, encode_and_escape(ie.file_id), encode_and_escape(ie.name), parent_str))
        elif ie.kind == 'directory':
            if not working:
                append(b'<directory file_id="%s" name="%s"%s revision="%s" />\n' % (encode_and_escape(ie.file_id), encode_and_escape(ie.name), parent_str, encode_and_escape(ie.revision)))
            else:
                append(b'<directory file_id="%s" name="%s"%s />\n' % (encode_and_escape(ie.file_id), encode_and_escape(ie.name), parent_str))
        elif ie.kind == 'symlink':
            if not working:
                append(b'<symlink file_id="%s" name="%s"%s revision="%s" symlink_target="%s" />\n' % (encode_and_escape(ie.file_id), encode_and_escape(ie.name), parent_str, encode_and_escape(ie.revision), encode_and_escape(ie.symlink_target)))
            else:
                append(b'<symlink file_id="%s" name="%s"%s />\n' % (encode_and_escape(ie.file_id), encode_and_escape(ie.name), parent_str))
        elif ie.kind == 'tree-reference':
            if ie.kind not in supported_kinds:
                raise serializer.UnsupportedInventoryKind(ie.kind)
            if not working:
                append(b'<tree-reference file_id="%s" name="%s"%s revision="%s" reference_revision="%s" />\n' % (encode_and_escape(ie.file_id), encode_and_escape(ie.name), parent_str, encode_and_escape(ie.revision), encode_and_escape(ie.reference_revision)))
            else:
                append(b'<tree-reference file_id="%s" name="%s"%s />\n' % (encode_and_escape(ie.file_id), encode_and_escape(ie.name), parent_str))
        else:
            raise serializer.UnsupportedInventoryKind(ie.kind)
    append(b'</inventory>\n')