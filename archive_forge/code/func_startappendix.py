import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def startappendix(self):
    """Start appendices here."""
    firsttype = self.orderedlayouts[DocumentParameters.startinglevel]
    counter = self.getcounter(firsttype)
    counter.setmode('A').reset()
    NumberGenerator.appendix = True