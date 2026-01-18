import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def checklimits(self, contents, index):
    """Check if the current position has a limits command."""
    if not DocumentParameters.displaymode:
        return False
    if self.checkcommand(contents, index + 1, LimitPreviousCommand):
        self.limitsahead(contents, index)
        return False
    if not isinstance(contents[index], LimitCommand):
        return False
    return self.checkscript(contents, index + 1)