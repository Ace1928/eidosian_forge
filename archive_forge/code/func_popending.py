import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def popending(self, expected=None):
    """Pop the ending found at the current position"""
    if self.isout() and self.leavepending:
        return expected
    ending = self.endinglist.pop(self)
    if expected and expected != ending:
        Trace.error('Expected ending ' + expected + ', got ' + ending)
    self.skip(ending)
    return ending