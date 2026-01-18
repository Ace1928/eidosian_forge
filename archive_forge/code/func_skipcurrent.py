import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def skipcurrent(self):
    """Return the current character and skip it."""
    current = self.current()
    self.skip(current)
    return current