import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def globexcluding(self, excluded):
    """Glob a bit of text up until (excluding) any excluded character."""
    return self.glob(lambda: self.current() not in excluded)