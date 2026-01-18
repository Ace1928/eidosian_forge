import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def processcontents(self, bit):
    """Process the contents of a formula bit."""
    if not isinstance(bit, FormulaBit):
        return
    bit.process()
    for element in bit.contents:
        self.processcontents(element)