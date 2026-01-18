import datetime
import sys
import threading
import time
import cherrypy
from cherrypy.lib import cptools, httputil
def tee_output():
    """Tee response output to cache storage. Internal."""
    request = cherrypy.serving.request
    if 'no-store' in request.headers.values('Cache-Control'):
        return

    def tee(body):
        """Tee response.body into a list."""
        if 'no-cache' in response.headers.values('Pragma') or 'no-store' in response.headers.values('Cache-Control'):
            for chunk in body:
                yield chunk
            return
        output = []
        for chunk in body:
            output.append(chunk)
            yield chunk
        body = b''.join(output)
        if not body:
            cherrypy._cache.delete()
        else:
            cherrypy._cache.put((response.status, response.headers or {}, body, response.time), len(body))
    response = cherrypy.serving.response
    response.body = tee(response.body)