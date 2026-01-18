import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def extractlstset(self, reader):
    """Extract the global lstset parameters."""
    paramtext = ''
    while not reader.finished():
        paramtext += reader.currentline()
        reader.nextline()
        if paramtext.endswith('}'):
            return paramtext
    Trace.error('Could not find end of \\lstset settings; aborting')