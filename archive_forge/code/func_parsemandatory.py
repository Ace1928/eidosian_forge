import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsemandatory(self, pos, number):
    """Parse a number of mandatory parameters."""
    for index in range(number):
        parameter = self.parsemacroparameter(pos, number - index)
        if not parameter:
            return
        self.values.append(parameter)