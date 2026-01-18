import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def computehybridsize(self):
    """Compute the size of the hybrid function."""
    if not self.command in HybridSize.configsizes:
        self.computesize()
        return
    self.size = HybridSize().getsize(self)
    for element in self.contents:
        element.size = self.size