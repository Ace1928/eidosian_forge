import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsealignments(self, pos):
    """Parse the different alignments"""
    self.valign = 'c'
    literal = self.parsesquareliteral(pos)
    if literal:
        self.valign = literal
    literal = self.parseliteral(pos)
    self.alignments = []
    for l in literal:
        self.alignments.append(l)