import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def jsmath(self):
    """Make the contents for jsMath."""
    if self.header[0] != 'inline':
        self.output = TaggedOutput().settag('div class="math"')
    else:
        self.output = TaggedOutput().settag('span class="math"')
    self.contents = [Constant(self.parsed)]