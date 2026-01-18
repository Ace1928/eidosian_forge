import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def checkskip(self, string):
    """Check for a string at the given position; if there, skip it"""
    if not self.checkfor(string):
        return False
    self.skip(string)
    return True