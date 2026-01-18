import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parselstparams(self, paramlist):
    """Process a number of lstparams from elyxer.a list."""
    paramdict = dict()
    for param in paramlist:
        if not '=' in param:
            if len(param.strip()) > 0:
                Trace.error('Invalid listing parameter ' + param)
        else:
            key, value = param.split('=', 1)
            paramdict[key] = value
    return paramdict