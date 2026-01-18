import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsetype(self, type, pos):
    """Parse the given type and return it."""
    bit = self.instance(type)
    self.instances[type] = None
    returnedbit = bit.parsebit(pos)
    if returnedbit:
        return returnedbit.setfactory(self)
    return bit