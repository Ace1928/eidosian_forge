import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parseending(self, reader, process):
    """Parse until the current ending is found"""
    if not self.ending:
        Trace.error('No ending for ' + str(self))
        return
    while not reader.currentline().startswith(self.ending):
        process()