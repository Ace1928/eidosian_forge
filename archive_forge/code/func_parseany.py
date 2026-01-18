import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parseany(self, pos):
    """Parse any formula bit at the current location."""
    for type in self.types + self.skippedtypes:
        if self.detecttype(type, pos):
            return self.parsetype(type, pos)
    Trace.error('Unrecognized formula at ' + pos.identifier())
    return FormulaConstant(pos.skipcurrent())