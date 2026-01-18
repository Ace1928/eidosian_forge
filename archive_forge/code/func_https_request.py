import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def https_request(self, request):
    return self.capture_connection(request, DavHTTPSConnection)