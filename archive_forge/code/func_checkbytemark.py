import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def checkbytemark(self):
    """Check for a Unicode byte mark and skip it."""
    if self.finished():
        return
    if ord(self.current()) == 65279:
        self.skipcurrent()