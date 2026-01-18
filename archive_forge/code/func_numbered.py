import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def numbered(self, container):
    """Get the numbered container for the label."""
    if container.partkey:
        return container
    if not container.parent:
        if self.lastnumbered:
            return self.lastnumbered
        return None
    return self.numbered(container.parent)