import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parselstset(self, reader):
    """Parse a declaration of lstparams in lstset."""
    paramtext = self.extractlstset(reader)
    if not '{' in paramtext:
        Trace.error('Missing opening bracket in lstset: ' + paramtext)
        return
    lefttext = paramtext.split('{')[1]
    croppedtext = lefttext[:-1]
    LstParser.globalparams = self.parselstparams(croppedtext)