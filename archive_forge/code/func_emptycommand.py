import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def emptycommand(self, pos):
    """Check for an empty command: look for command disguised as ending.
    Special case against '{ \\{ \\} }' situation."""
    command = ''
    if not pos.isout():
        ending = pos.nextending()
        if ending and pos.checkskip(ending):
            command = ending
    return FormulaCommand.start + command