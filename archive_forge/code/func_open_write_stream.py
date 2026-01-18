import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def open_write_stream(self, relpath, mode=None):
    """See Transport.open_write_stream."""
    self.put_bytes(relpath, b'', mode)
    result = transport.AppendBasedFileStream(self, relpath)
    transport._file_streams[self.abspath(relpath)] = result
    return result