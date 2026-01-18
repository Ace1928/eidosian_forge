import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def startDocument(self):
    self.elt_stack = []
    self.chars = None
    self.expected_content_handled = False