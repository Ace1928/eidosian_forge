import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def isout(self):
    """Find out if we are out of the text yet."""
    if self.pos > len(self.reader.currentline()):
        if self.pos > len(self.reader.currentline()) + 1:
            Trace.error('Out of the line ' + self.reader.currentline() + ': ' + str(self.pos))
        self.nextline()
    return self.reader.finished()