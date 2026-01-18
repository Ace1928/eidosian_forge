import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def endElement(self, name):
    if self._response_seen:
        self._additional_response_starting(name)
    if self._href_end():
        self.href = self.chars
    elif self._getcontentlength_end():
        self.length = int(self.chars)
    elif self._executable_end():
        self.executable = self.chars
    elif self._collection_end():
        self.is_dir = True
    if self._strip_ns(name) == 'response':
        self._response_seen = True
        self._response_handled()
    DavResponseHandler.endElement(self, name)