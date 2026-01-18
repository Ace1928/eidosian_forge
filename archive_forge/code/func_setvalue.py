import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def setvalue(self, name, value):
    """Set the value of a parameter name, only if it's valid."""
    value = self.processparameter(value)
    if value:
        setattr(self, name, value)