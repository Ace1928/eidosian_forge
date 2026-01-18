import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getpiece4(self, index):
    """Get the nth piece for a 4-piece bracket: curly bracket."""
    if index == 0:
        return self.pieces[0]
    if index == self.size - 1:
        return self.pieces[3]
    if index == (self.size - 1) / 2:
        return self.pieces[2]
    return self.pieces[1]