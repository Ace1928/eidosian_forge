import sys
import os
import re
import io
import shutil
import socket
import base64
import hashlib
import itertools
import configparser
import html
import http.client
import urllib.parse
import urllib.request
import urllib.error
from functools import wraps
import setuptools
from pkg_resources import (
from distutils import log
from distutils.errors import DistutilsError
from fnmatch import translate
from setuptools.wheel import Wheel
from setuptools.extern.more_itertools import unique_everseen
def open_url(self, url, warning=None):
    if url.startswith('file:'):
        return local_open(url)
    try:
        return open_with_auth(url, self.opener)
    except (ValueError, http.client.InvalidURL) as v:
        msg = ' '.join([str(arg) for arg in v.args])
        if warning:
            self.warn(warning, msg)
        else:
            raise DistutilsError('%s %s' % (url, msg)) from v
    except urllib.error.HTTPError as v:
        return v
    except urllib.error.URLError as v:
        if warning:
            self.warn(warning, v.reason)
        else:
            raise DistutilsError('Download error for %s: %s' % (url, v.reason)) from v
    except http.client.BadStatusLine as v:
        if warning:
            self.warn(warning, v.line)
        else:
            raise DistutilsError('%s returned a bad status line. The server might be down, %s' % (url, v.line)) from v
    except (http.client.HTTPException, socket.error) as v:
        if warning:
            self.warn(warning, v)
        else:
            raise DistutilsError('Download error for %s: %s' % (url, v)) from v