import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def iteraterows(self, pos):
    """Iterate over all rows, end when no more row ends"""
    rowseparator = FormulaConfig.array['rowseparator']
    while True:
        pos.pushending(rowseparator, True)
        row = self.factory.create(FormulaRow)
        yield row.setalignments(self.alignments)
        if pos.checkfor(rowseparator):
            self.original += pos.popending(rowseparator)
        else:
            return