import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsecontainer(self, container):
    """Parse some lstparams from elyxer.a container."""
    container.lstparams = LstParser.globalparams.copy()
    paramlist = container.getparameterlist('lstparams')
    container.lstparams.update(self.parselstparams(paramlist))