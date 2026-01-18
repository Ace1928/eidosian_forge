import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def setparameter(self, container, name):
    """Read a size parameter off a container, and set it if present."""
    value = container.getparameter(name)
    self.setvalue(name, value)