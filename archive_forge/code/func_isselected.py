import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def isselected(self):
    """Return if the branch is selected"""
    if not 'selected' in self.options:
        return False
    return self.options['selected'] == '1'