import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def endDocument(self):
    self._validate_handling()
    if not self.expected_content_handled:
        raise errors.InvalidHttpResponse(self.url, msg='Unknown xml response')