import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def findending(self, pos):
    """Find the ending at the current position"""
    if len(self.endings) == 0:
        return None
    for index, ending in enumerate(reversed(self.endings)):
        if ending.checkin(pos):
            return ending
        if not ending.optional:
            return None
    return None