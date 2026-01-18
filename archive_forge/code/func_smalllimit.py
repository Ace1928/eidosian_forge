import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def smalllimit(self):
    """Decide if the limit should be a small, one-line symbol."""
    if not DocumentParameters.displaymode:
        return True
    if len(self.symbols[self.symbol]) == 1:
        return True
    return Options.simplemath