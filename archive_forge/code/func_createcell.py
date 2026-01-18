import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def createcell(self, index):
    """Create the cell that corresponds to the given index."""
    alignment = self.alignments[index % len(self.alignments)]
    return self.factory.create(FormulaCell).setalignment(alignment)