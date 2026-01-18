import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getparam(self, name):
    """Get a parameter as parsed."""
    if not name in self.params:
        return None
    return self.params[name]