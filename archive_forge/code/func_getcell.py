import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getcell(self, index):
    """Get the bracket piece as an array cell."""
    piece = self.getpiece(index)
    span = 'span class="bracket align-' + self.alignment + '"'
    return TaggedBit().constant(piece, span)