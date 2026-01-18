import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
def autovary(ignore=None, debug=False):
    """Auto-populate the Vary response header based on request.header access.
    """
    request = cherrypy.serving.request
    req_h = request.headers
    request.headers = MonitoredHeaderMap()
    request.headers.update(req_h)
    if ignore is None:
        ignore = set(['Content-Disposition', 'Content-Length', 'Content-Type'])

    def set_response_header():
        resp_h = cherrypy.serving.response.headers
        v = set([e.value for e in resp_h.elements('Vary')])
        if debug:
            cherrypy.log('Accessed headers: %s' % request.headers.accessed_headers, 'TOOLS.AUTOVARY')
        v = v.union(request.headers.accessed_headers)
        v = v.difference(ignore)
        v = list(v)
        v.sort()
        resp_h['Vary'] = ', '.join(v)
    request.hooks.attach('before_finalize', set_response_header, 95)