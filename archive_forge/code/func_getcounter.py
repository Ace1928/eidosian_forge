import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getcounter(self, type):
    """Get the counter for the given type."""
    type = type.lower()
    if not type in self.counters:
        self.counters[type] = self.create(type)
    return self.counters[type]