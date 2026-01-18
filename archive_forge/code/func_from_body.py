from io import StringIO
from mimetypes import MimeTypes
from pkgutil import get_data
from typing import Dict, Mapping, Optional, Type, Union
from scrapy.http import Response
from scrapy.utils.misc import load_object
from scrapy.utils.python import binary_is_text, to_bytes, to_unicode
def from_body(self, body: bytes) -> Type[Response]:
    """Try to guess the appropriate response based on the body content.
        This method is a bit magic and could be improved in the future, but
        it's not meant to be used except for special cases where response types
        cannot be guess using more straightforward methods."""
    chunk = body[:5000]
    chunk = to_bytes(chunk)
    if not binary_is_text(chunk):
        return self.from_mimetype('application/octet-stream')
    lowercase_chunk = chunk.lower()
    if b'<html>' in lowercase_chunk:
        return self.from_mimetype('text/html')
    if b'<?xml' in lowercase_chunk:
        return self.from_mimetype('text/xml')
    if b'<!doctype html>' in lowercase_chunk:
        return self.from_mimetype('text/html')
    return self.from_mimetype('text')