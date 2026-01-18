import re
import sys
import tempfile
from urllib.parse import unquote
import cheroot.server
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
def fullvalue(self):
    """Return this entity as a string, whether stored in a file or not."""
    if self.file:
        self.file.seek(0)
        value = self.file.read()
        self.file.seek(0)
    else:
        value = self.value
    value = self.decode_entity(value)
    return value