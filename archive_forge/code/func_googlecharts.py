import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def googlecharts(self):
    """Make the contents using Google Charts http://code.google.com/apis/chart/."""
    url = FormulaConfig.urls['googlecharts'] + urllib.parse.quote_plus(self.parsed)
    img = '<img class="chart" src="' + url + '" alt="' + self.parsed + '"/>'
    self.contents = [Constant(img)]