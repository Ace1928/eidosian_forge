import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def buddy(self, widget, prop):
    buddy_name = prop[0].text
    if buddy_name:
        self.buddies.append((widget, buddy_name))