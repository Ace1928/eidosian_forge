import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getlimit(self, contents, index):
    """Get the limit for a limits command."""
    limit = self.getscript(contents, index)
    limit.output.tag = limit.output.tag.replace('script', 'limit')
    return limit