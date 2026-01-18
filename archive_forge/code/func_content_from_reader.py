import codecs
import functools
import json
import os
import traceback
from testtools.compat import _b
from testtools.content_type import ContentType, JSON, UTF8_TEXT
def content_from_reader(reader, content_type, buffer_now):
    """Create a Content object that will obtain the content from reader.

    :param reader: A callback to read the content. Should return an iterable of
        bytestrings.
    :param content_type: The content type to create.
    :param buffer_now: If True the reader is evaluated immediately and
        buffered.
    """
    if content_type is None:
        content_type = UTF8_TEXT
    if buffer_now:
        contents = list(reader())

        def reader():
            return contents
    return Content(content_type, reader)