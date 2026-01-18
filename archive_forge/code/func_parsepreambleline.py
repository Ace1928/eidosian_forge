import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsepreambleline(self, reader):
    """Parse a single preamble line."""
    PreambleParser.preamble.append(reader.currentline())
    reader.nextline()