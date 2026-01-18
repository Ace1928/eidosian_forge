import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def processinsides(self, bit):
    """Process the insides (limits, brackets) in a formula bit."""
    if not isinstance(bit, FormulaBit):
        return
    for index, element in enumerate(bit.contents):
        for processor in self.processors:
            processor.process(bit.contents, index)
        self.processinsides(element)