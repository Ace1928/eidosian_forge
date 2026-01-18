import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def checktag(self):
    """Check that the tag is valid."""
    if not self.tag:
        Trace.error('No tag in ' + str(container))
        return False
    if self.tag == '':
        return False
    return True