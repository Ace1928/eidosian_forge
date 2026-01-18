import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def detecttype(self, type, pos):
    """Detect a bit of a given type."""
    if pos.finished():
        return False
    return self.instance(type).detect(pos)