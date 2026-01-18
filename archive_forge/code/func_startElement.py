import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def startElement(self, name, attrs):
    sname = self._strip_ns(name)
    self.chars_wanted = sname in ('href', 'getcontentlength', 'executable')
    DavResponseHandler.startElement(self, name, attrs)