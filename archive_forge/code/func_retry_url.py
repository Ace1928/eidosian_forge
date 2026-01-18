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
def retry_url(url, retry_on_404=True, num_retries=10, timeout=None):
    """
    Retry a url.  This is specifically used for accessing the metadata
    service on an instance.  Since this address should never be proxied
    (for security reasons), we create a ProxyHandler with a NULL
    dictionary to override any proxy settings in the environment.
    """
    for i in range(0, num_retries):
        try:
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            req = urllib.request.Request(url)
            r = opener.open(req, timeout=timeout)
            result = r.read()
            if not isinstance(result, six.string_types) and hasattr(result, 'decode'):
                result = result.decode('utf-8')
            return result
        except urllib.error.HTTPError as e:
            code = e.getcode()
            if code == 404 and (not retry_on_404):
                return ''
        except Exception as e:
            boto.log.exception('Caught exception reading instance data')
        if i + 1 != num_retries:
            boto.log.debug('Sleeping before retrying')
            time.sleep(min(2 ** i, boto.config.get('Boto', 'max_retry_delay', 60)))
    boto.log.error('Unable to read instance data, giving up')
    return ''