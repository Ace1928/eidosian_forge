import copy
import os
import sys
from io import BytesIO
from xml.dom.minidom import getDOMImplementation
from twisted.internet import address, reactor
from twisted.logger import Logger
from twisted.persisted import styles
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.web import http, resource, server, static, util
from twisted.web.http_headers import Headers
def remote_request(self, request):
    """
        Look up the resource for the given request and render it.
        """
    res = self.site.getResourceFor(request)
    self._log.info(request)
    result = res.render(request)
    if result is not server.NOT_DONE_YET:
        request.write(result)
        request.finish()
    return server.NOT_DONE_YET