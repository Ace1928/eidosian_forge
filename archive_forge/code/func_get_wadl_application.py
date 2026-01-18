import atexit
import errno
import os
import re
import shutil
import sys
import tempfile
from hashlib import md5
from io import BytesIO
from json import dumps
from time import sleep
from httplib2 import Http, urlnorm
from wadllib.application import Application
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError, error_for
from lazr.uri import URI
def get_wadl_application(self, url):
    """GET a WADL representation of the resource at the requested url."""
    wadl_type = 'application/vnd.sun.wadl+xml'
    response, content = self._request(url, media_type=wadl_type)
    url = str(url)
    if not isinstance(content, bytes):
        content = content.encode('utf-8')
    return Application(url, content)