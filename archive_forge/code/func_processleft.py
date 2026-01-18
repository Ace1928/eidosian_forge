import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def processleft(self, contents, index):
    """Process a left bracket."""
    rightindex = self.findright(contents, index + 1)
    if not rightindex:
        return
    size = self.findmax(contents, index, rightindex)
    self.resize(contents[index], size)
    self.resize(contents[rightindex], size)