import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getdependentcounter(self, type, master):
    """Get (or create) a counter of the given type that depends on another."""
    if not type in self.counters or not self.counters[type].master:
        self.counters[type] = self.createdependent(type, master)
    return self.counters[type]