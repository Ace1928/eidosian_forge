from io import StringIO, BytesIO, TextIOWrapper
from collections.abc import Mapping
import sys
import os
import urllib.parse
from email.parser import FeedParser
from email.message import Message
import html
import locale
import tempfile
import warnings
def read_multi(self, environ, keep_blank_values, strict_parsing):
    """Internal: read a part that is itself multipart."""
    ib = self.innerboundary
    if not valid_boundary(ib):
        raise ValueError('Invalid boundary in multipart form: %r' % (ib,))
    self.list = []
    if self.qs_on_post:
        query = urllib.parse.parse_qsl(self.qs_on_post, self.keep_blank_values, self.strict_parsing, encoding=self.encoding, errors=self.errors, max_num_fields=self.max_num_fields, separator=self.separator)
        self.list.extend((MiniFieldStorage(key, value) for key, value in query))
    klass = self.FieldStorageClass or self.__class__
    first_line = self.fp.readline()
    if not isinstance(first_line, bytes):
        raise ValueError('%s should return bytes, got %s' % (self.fp, type(first_line).__name__))
    self.bytes_read += len(first_line)
    while first_line.strip() != b'--' + self.innerboundary and first_line:
        first_line = self.fp.readline()
        self.bytes_read += len(first_line)
    max_num_fields = self.max_num_fields
    if max_num_fields is not None:
        max_num_fields -= len(self.list)
    while True:
        parser = FeedParser()
        hdr_text = b''
        while True:
            data = self.fp.readline()
            hdr_text += data
            if not data.strip():
                break
        if not hdr_text:
            break
        self.bytes_read += len(hdr_text)
        parser.feed(hdr_text.decode(self.encoding, self.errors))
        headers = parser.close()
        if 'content-length' in headers:
            del headers['content-length']
        limit = None if self.limit is None else self.limit - self.bytes_read
        part = klass(self.fp, headers, ib, environ, keep_blank_values, strict_parsing, limit, self.encoding, self.errors, max_num_fields, self.separator)
        if max_num_fields is not None:
            max_num_fields -= 1
            if part.list:
                max_num_fields -= len(part.list)
            if max_num_fields < 0:
                raise ValueError('Max number of fields exceeded')
        self.bytes_read += part.bytes_read
        self.list.append(part)
        if part.done or self.bytes_read >= self.length > 0:
            break
    self.skip_lines()