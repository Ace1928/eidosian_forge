import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def setalignment(self, alignment):
    self.alignment = alignment
    self.output = TaggedOutput().settag('span class="arraycell align-' + alignment + '"', True)
    return self