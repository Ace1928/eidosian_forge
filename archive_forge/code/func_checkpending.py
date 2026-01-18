import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def checkpending(self):
    """Check if there are any pending endings"""
    if len(self.endings) != 0:
        Trace.error('Pending ' + str(self) + ' left open')