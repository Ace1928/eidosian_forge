import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def searchremove(self, type):
    """Search for all containers of a type and remove them"""
    list = self.searchall(type)
    for container in list:
        container.parent.contents.remove(container)
    return list