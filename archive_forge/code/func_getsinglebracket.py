import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getsinglebracket(self):
    """Return the bracket as a single sign."""
    if self.original == '.':
        return [TaggedBit().constant('', 'span class="emptydot"')]
    return [TaggedBit().constant(self.original, 'span class="symbol"')]