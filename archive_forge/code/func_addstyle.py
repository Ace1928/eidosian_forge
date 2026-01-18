import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def addstyle(self, container):
    """Add the proper style attribute to the output tag."""
    if not isinstance(container.output, TaggedOutput):
        Trace.error('No tag to add style, in ' + str(container))
    if not self.width and (not self.height) and (not self.maxwidth) and (not self.maxheight):
        return
    tag = ' style="'
    tag += self.styleparameter('width')
    tag += self.styleparameter('maxwidth')
    tag += self.styleparameter('height')
    tag += self.styleparameter('maxheight')
    if tag[-1] == ' ':
        tag = tag[:-1]
    tag += '"'
    container.output.tag += tag