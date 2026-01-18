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
def get_instance_identity(version='latest', url='http://169.254.169.254', timeout=None, num_retries=5):
    """
    Returns the instance identity as a nested Python dictionary.
    """
    iid = {}
    base_url = _build_instance_metadata_url(url, version, 'dynamic/instance-identity/')
    try:
        data = retry_url(base_url, num_retries=num_retries, timeout=timeout)
        fields = data.split('\n')
        for field in fields:
            val = retry_url(base_url + '/' + field + '/', num_retries=num_retries, timeout=timeout)
            if val[0] == '{':
                val = json.loads(val)
            if field:
                iid[field] = val
        return iid
    except urllib.error.URLError:
        return None