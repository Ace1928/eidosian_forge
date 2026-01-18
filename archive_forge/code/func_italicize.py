import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def italicize(self, bit, contents):
    """Italicize the given bit of text."""
    index = contents.index(bit)
    contents[index] = TaggedBit().complete([bit], 'i')