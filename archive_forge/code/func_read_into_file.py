import re
import sys
import tempfile
from urllib.parse import unquote
import cheroot.server
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
def read_into_file(self, fp_out=None):
    """Read the request body into fp_out (or make_file() if None).

        Return fp_out.
        """
    if fp_out is None:
        fp_out = self.make_file()
    self.read_lines_to_boundary(fp_out=fp_out)
    return fp_out