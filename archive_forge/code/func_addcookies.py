from suds.properties import Unskin
from suds.transport import *
import base64
from http.cookiejar import CookieJar
import http.client
import socket
import sys
import urllib.request, urllib.error, urllib.parse
import gzip
import zlib
from logging import getLogger
def addcookies(self, u2request):
    """
        Add cookies in the cookiejar to the request.

        @param u2request: A urllib2 request.
        @rtype: u2request: urllib2.Request.

        """
    self.cookiejar.add_cookie_header(u2request)