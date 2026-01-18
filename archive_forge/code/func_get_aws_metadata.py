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
def get_aws_metadata(headers, provider=None):
    if not provider:
        provider = boto.provider.get_default()
    metadata_prefix = provider.metadata_prefix
    metadata = {}
    for hkey in headers.keys():
        if hkey.lower().startswith(metadata_prefix):
            val = urllib.parse.unquote(headers[hkey])
            if isinstance(val, bytes):
                try:
                    val = val.decode('utf-8')
                except UnicodeDecodeError:
                    pass
            metadata[hkey[len(metadata_prefix):]] = val
            del headers[hkey]
    return metadata