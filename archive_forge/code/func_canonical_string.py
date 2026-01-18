import subprocess
import time
import logging.handlers
import boto
import boto.provider
import collections
import tempfile
import random
import smtplib
import datetime
import re
import io
import email.mime.multipart
import email.mime.base
import email.mime.text
import email.utils
import email.encoders
import gzip
import threading
import locale
import sys
from boto.compat import six, StringIO, urllib, encodebytes
from contextlib import contextmanager
from hashlib import md5, sha512
from boto.compat import json
def canonical_string(method, path, headers, expires=None, provider=None):
    """
    Generates the aws canonical string for the given parameters
    """
    if not provider:
        provider = boto.provider.get_default()
    interesting_headers = {}
    for key in headers:
        lk = key.lower()
        if headers[key] is not None and (lk in ['content-md5', 'content-type', 'date'] or lk.startswith(provider.header_prefix)):
            interesting_headers[lk] = str(headers[key]).strip()
    if 'content-type' not in interesting_headers:
        interesting_headers['content-type'] = ''
    if 'content-md5' not in interesting_headers:
        interesting_headers['content-md5'] = ''
    if provider.date_header in interesting_headers:
        interesting_headers['date'] = ''
    if expires:
        interesting_headers['date'] = str(expires)
    sorted_header_keys = sorted(interesting_headers.keys())
    buf = '%s\n' % method
    for key in sorted_header_keys:
        val = interesting_headers[key]
        if key.startswith(provider.header_prefix):
            buf += '%s:%s\n' % (key, val)
        else:
            buf += '%s\n' % val
    t = path.split('?')
    buf += t[0]
    if len(t) > 1:
        qsa = t[1].split('&')
        qsa = [a.split('=', 1) for a in qsa]
        qsa = [unquote_v(a) for a in qsa if a[0] in qsa_of_interest]
        if len(qsa) > 0:
            qsa.sort(key=lambda x: x[0])
            qsa = ['='.join(a) for a in qsa]
            buf += '?'
            buf += '&'.join(qsa)
    return buf