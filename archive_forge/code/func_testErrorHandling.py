from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
def testErrorHandling(self):
    self.getPage('/error/missing')
    self.assertStatus(404)
    self.assertErrorPage(404, "The path '/error/missing' was not found.")
    ignore = helper.webtest.ignored_exceptions
    ignore.append(ValueError)
    try:
        valerr = '\n    raise ValueError()\nValueError'
        self.getPage('/error/page_method')
        self.assertErrorPage(500, pattern=valerr)
        self.getPage('/error/page_yield')
        self.assertErrorPage(500, pattern=valerr)
        if cherrypy.server.protocol_version == 'HTTP/1.0' or getattr(cherrypy.server, 'using_apache', False):
            self.getPage('/error/page_streamed')
            self.assertStatus(200)
            self.assertBody('word up')
        else:
            self.assertRaises((ValueError, IncompleteRead), self.getPage, '/error/page_streamed')
        self.getPage('/error/cause_err_in_finalize')
        msg = "Illegal response status from server ('ZOO' is non-numeric)."
        self.assertErrorPage(500, msg, None)
    finally:
        ignore.pop()
    self.getPage('/error/reason_phrase')
    self.assertStatus("410 Gone fishin'")
    self.getPage('/error/custom')
    self.assertStatus(404)
    self.assertBody('Hello, world\r\n' + ' ' * 499)
    self.getPage('/error/custom?err=401')
    self.assertStatus(401)
    self.assertBody("Error 401 Unauthorized - Well, I'm very sorry but you haven't paid!")
    self.getPage('/error/custom_default')
    self.assertStatus(500)
    self.assertBody("Error 500 Internal Server Error - Well, I'm very sorry but you haven't paid!".ljust(513))
    self.getPage('/error/noexist')
    self.assertStatus(404)
    if sys.version_info >= (3, 3):
        exc_name = 'FileNotFoundError'
    else:
        exc_name = 'IOError'
    msg = "No, &lt;b&gt;really&lt;/b&gt;, not found!<br />In addition, the custom error page failed:\n<br />%s: [Errno 2] No such file or directory: 'nonexistent.html'" % (exc_name,)
    self.assertInBody(msg)
    if getattr(cherrypy.server, 'using_apache', False):
        pass
    else:
        self.getPage('/error/rethrow')
        self.assertInBody('raise ValueError()')