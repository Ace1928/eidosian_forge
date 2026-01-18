import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def hasemptyoutput(self):
    """Check if the parent's output is empty."""
    current = self.parent
    while current:
        if current.output.isempty():
            return True
        current = current.parent
    return False