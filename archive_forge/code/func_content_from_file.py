import codecs
import functools
import json
import os
import traceback
from testtools.compat import _b
from testtools.content_type import ContentType, JSON, UTF8_TEXT
def content_from_file(path, content_type=None, chunk_size=DEFAULT_CHUNK_SIZE, buffer_now=False, seek_offset=None, seek_whence=0):
    """Create a Content object from a file on disk.

    Note that unless ``buffer_now`` is explicitly passed in as True, the file
    will only be read from when ``iter_bytes`` is called.

    :param path: The path to the file to be used as content.
    :param content_type: The type of content.  If not specified, defaults
        to UTF8-encoded text/plain.
    :param chunk_size: The size of chunks to read from the file.
        Defaults to ``DEFAULT_CHUNK_SIZE``.
    :param buffer_now: If True, read the file from disk now and keep it in
        memory. Otherwise, only read when the content is serialized.
    :param seek_offset: If non-None, seek within the stream before reading it.
    :param seek_whence: If supplied, pass to ``stream.seek()`` when seeking.
    """
    if content_type is None:
        content_type = UTF8_TEXT

    def reader():
        with open(path, 'rb') as stream:
            yield from _iter_chunks(stream, chunk_size, seek_offset, seek_whence)
    return content_from_reader(reader, content_type, buffer_now)