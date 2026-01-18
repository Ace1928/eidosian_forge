import datetime
from itertools import count
import os
import threading
import time
import urllib.parse
import pytest
import cherrypy
from cherrypy.lib import httputil
from cherrypy.test import helper
def testGzipStaticCache(self):
    """Test that cache and gzip tools play well together when both enabled.

        Ref GitHub issue #1190.
        """
    GZIP_STATIC_CACHE_TMPL = '/gzip_static_cache/{}'
    resource_files = ('index.html', 'dirback.jpg')
    for f in resource_files:
        uri = GZIP_STATIC_CACHE_TMPL.format(f)
        self._assert_resp_len_and_enc_for_gzip(uri)