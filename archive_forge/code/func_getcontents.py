import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getcontents(self):
    """Get the bracket as an array or as a single bracket."""
    if self.size == 1 or not self.pieces:
        return self.getsinglebracket()
    rows = []
    for index in range(self.size):
        cell = self.getcell(index)
        rows.append(TaggedBit().complete([cell], 'span class="arrayrow"'))
    return [TaggedBit().complete(rows, 'span class="array"')]