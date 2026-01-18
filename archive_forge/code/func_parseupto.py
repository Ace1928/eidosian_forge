import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parseupto(self, pos, limit):
    """Parse a formula that ends with the given command."""
    pos.pushending(limit)
    self.parsed = pos.glob(lambda: True)
    pos.popending(limit)