import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def checkcommand(self, contents, index, type):
    """Check for the given type as the current element."""
    if len(contents) <= index:
        return False
    return isinstance(contents[index], type)