import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def setcommand(self, command):
    """Set the command in the bit"""
    self.command = command
    if self.commandmap:
        self.original += command
        self.translated = self.commandmap[self.command]