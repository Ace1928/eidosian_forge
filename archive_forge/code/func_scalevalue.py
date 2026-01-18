import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def scalevalue(self, value):
    """Scale the value according to the image scale and return it as unicode."""
    scaled = value * int(self.scale) / 100
    return str(int(scaled)) + 'px'