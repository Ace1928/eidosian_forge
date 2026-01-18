import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def locateprocess(self, locate, process):
    """Search for all embedded containers and process them"""
    for container in self.contents:
        container.locateprocess(locate, process)
        if locate(container):
            process(container)