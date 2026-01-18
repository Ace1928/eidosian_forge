import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsemacroparameter(self, pos, remaining):
    """Parse a macro parameter. Could be a bracket or a single letter."""
    'If there are just two values remaining and there is a running number,'
    'parse as two separater numbers.'
    self.factory.clearskipped(pos)
    if pos.finished():
        return None
    if self.factory.detecttype(FormulaNumber, pos):
        return self.parsenumbers(pos, remaining)
    return self.parseparameter(pos)