import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def computedestination(self):
    """Use the destination link to fill in the destination URL."""
    if not self.destination:
        return
    self.url = ''
    if self.destination.anchor:
        self.url = '#' + self.destination.anchor
    if self.destination.page:
        self.url = self.destination.page + self.url