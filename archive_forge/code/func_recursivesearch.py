import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def recursivesearch(self, locate, recursive, process):
    """Perform a recursive search in the container."""
    for container in self.contents:
        if recursive(container):
            container.recursivesearch(locate, recursive, process)
        if locate(container):
            process(container)