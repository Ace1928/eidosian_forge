import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def checkimage(self, width, height):
    """Check image dimensions, set them if possible."""
    if width:
        self.maxwidth = str(width) + 'px'
        if self.scale and (not self.width):
            self.width = self.scalevalue(width)
    if height:
        self.maxheight = str(height) + 'px'
        if self.scale and (not self.height):
            self.height = self.scalevalue(height)
    if self.width and (not self.height):
        self.height = 'auto'
    if self.height and (not self.width):
        self.width = 'auto'