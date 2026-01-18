import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def addsymbol(self, symbol, pos):
    """Add a symbol"""
    self.skiporiginal(pos.current(), pos)
    self.contents.append(FormulaConstant(symbol))