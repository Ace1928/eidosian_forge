import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def checkbadchars(url):
    proto, uri = url.split('://', 1)
    if proto != 'file':
        host, uripath = uri.split('/', 1)
        if _check_for_bad_chars(host, ALLOWED_CHARS_HOST) or _check_for_bad_chars(uripath, ALLOWED_CHARS):
            raise ValueError('bad char in %r' % (url,))