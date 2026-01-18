import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsesquare(self, pos):
    """Parse a square bracket"""
    self.factory.clearskipped(pos)
    if not self.factory.detecttype(SquareBracket, pos):
        return None
    bracket = self.factory.parsetype(SquareBracket, pos)
    self.add(bracket)
    return bracket