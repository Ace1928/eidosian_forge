import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsecomplete(self, pos, innerparser):
    """Parse the start and end marks"""
    if not pos.checkfor(self.start):
        Trace.error('Bracket should start with ' + self.start + ' at ' + pos.identifier())
        return None
    self.skiporiginal(self.start, pos)
    pos.pushending(self.ending)
    innerparser(pos)
    self.original += pos.popending(self.ending)
    self.computesize()