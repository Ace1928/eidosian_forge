import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def checkleft(self, contents, index):
    """Check if the command at the given index is left."""
    return self.checkdirection(contents[index], '\\left')