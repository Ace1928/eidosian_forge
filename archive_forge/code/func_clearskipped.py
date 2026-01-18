import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def clearskipped(self, pos):
    """Clear any skipped types."""
    while not pos.finished():
        if not self.skipany(pos):
            return
    return