import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def setstart(self, firstline):
    """Set the first line to read."""
    for i in range(firstline):
        self.file.readline()
    self.linenumber = firstline