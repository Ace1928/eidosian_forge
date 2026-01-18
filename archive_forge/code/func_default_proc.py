import re
import sys
import tempfile
from urllib.parse import unquote
import cheroot.server
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
def default_proc(self):
    """Called if a more-specific processor is not found for the
        ``Content-Type``.
        """
    if self.filename:
        self.file = self.read_into_file()
    else:
        result = self.read_lines_to_boundary()
        if isinstance(result, bytes):
            self.value = result
        else:
            self.file = result