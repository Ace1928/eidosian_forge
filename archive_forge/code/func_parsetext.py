import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsetext(self, pos):
    """Parse a text parameter."""
    self.factory.clearskipped(pos)
    if not self.factory.detecttype(Bracket, pos):
        Trace.error('No text parameter for ' + self.command)
        return None
    bracket = Bracket().setfactory(self.factory).parsetext(pos)
    self.add(bracket)
    return bracket