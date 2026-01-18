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
def get_utf8able_str(s, errors='strict'):
    """Returns a UTF8-encodable string in PY3, UTF8 bytes in PY2.

    This method is similar to six's `ensure_str()`, except it also
    makes sure that any bytes passed in can be decoded using the
    utf-8 codec (and raises a UnicodeDecodeError if not). If the
    object isn't a string, this method will attempt to coerce it
    to a string with `str()`. Objects without `__str__` property
    or `__repr__` property will raise an exception.
    """
    if not isinstance(s, (six.text_type, six.binary_type)):
        s = str(s)
    if six.PY2:
        if isinstance(s, six.text_type):
            return s.encode('utf-8', errors)
        if isinstance(s, six.binary_type):
            s.decode('utf-8')
            return s
    else:
        if isinstance(s, six.text_type):
            return s
        if isinstance(s, six.binary_type):
            s = s.decode('utf-8')
            return s
    raise TypeError('not expecting type "%s"' % type(s))