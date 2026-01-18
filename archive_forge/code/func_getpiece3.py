import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getpiece3(self, index):
    """Get the nth piece for a 3-piece bracket: parenthesis or square bracket."""
    if index == 0:
        return self.pieces[0]
    if index == self.size - 1:
        return self.pieces[-1]
    return self.pieces[1]