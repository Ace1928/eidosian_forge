import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def findmax(self, contents, leftindex, rightindex):
    """Find the max size of the contents between the two given indices."""
    sliced = contents[leftindex:rightindex]
    return max([element.size for element in sliced])