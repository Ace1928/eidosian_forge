import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getstage(self, element):
    """Get the stage for a given element, if the type is in the dict"""
    if not element.__class__ in self.stagedict:
        return None
    return self.stagedict[element.__class__]