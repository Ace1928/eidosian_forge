import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parserows(self, pos):
    """Parse all rows, finish when no more row ends"""
    self.rows = []
    first = True
    for row in self.iteraterows(pos):
        if first:
            first = False
        else:
            self.addempty()
        row.parsebit(pos)
        self.addrow(row)
    self.size = len(self.rows)