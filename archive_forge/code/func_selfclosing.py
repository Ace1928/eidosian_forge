import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def selfclosing(self, container):
    """Get self-closing line."""
    if not self.checktag():
        return ''
    selfclosing = '<' + self.tag + '/>'
    if self.breaklines:
        return selfclosing + '\n'
    return selfclosing