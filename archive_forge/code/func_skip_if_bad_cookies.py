import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def skip_if_bad_cookies(self):
    """
        cookies module fails to reject invalid cookies
        https://github.com/cherrypy/cherrypy/issues/1405
        """
    cookies = sys.modules.get('http.cookies')
    _is_legal_key = getattr(cookies, '_is_legal_key', lambda x: False)
    if not _is_legal_key(','):
        return
    issue = 'http://bugs.python.org/issue26302'
    tmpl = 'Broken cookies module ({issue})'
    self.skip(tmpl.format(**locals()))