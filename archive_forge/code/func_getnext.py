import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getnext(self):
    """Increase or, if the master counter has changed, restart."""
    if self.last != self.master.getvalue():
        self.reset()
    value = NumberCounter.getnext(self)
    self.last = self.master.getvalue()
    return value