import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def limitsahead(self, contents, index):
    """Limit the current element based on the next."""
    contents[index + 1].add(contents[index].clone())
    contents[index].output = EmptyOutput()