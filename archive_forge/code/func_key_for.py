import os
import win32api
import win32con
import win32event
import win32service
import win32serviceutil
from cherrypy.process import wspbus, plugins
def key_for(self, obj):
    """For the given value, return its corresponding key."""
    for key, val in self.items():
        if val is obj:
            return key
    raise ValueError('The given object could not be found: %r' % obj)