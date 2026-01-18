import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parseformula(self, formula):
    """Parse a string of text that contains a whole formula."""
    pos = TextPosition(formula)
    whole = self.create(WholeFormula)
    if whole.detect(pos):
        whole.parsebit(pos)
        return whole
    if not pos.finished():
        Trace.error('Unknown formula at: ' + pos.identifier())
        whole.add(TaggedBit().constant(formula, 'span class="unknown"'))
    return whole