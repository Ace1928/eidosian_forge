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
def get_instance_userdata(version='latest', sep=None, url='http://169.254.169.254', timeout=None, num_retries=5):
    ud_url = _build_instance_metadata_url(url, version, 'user-data')
    user_data = retry_url(ud_url, retry_on_404=False, num_retries=num_retries, timeout=timeout)
    if user_data:
        if sep:
            l = user_data.split(sep)
            user_data = {}
            for nvpair in l:
                t = nvpair.split('=')
                user_data[t[0].strip()] = t[1].strip()
    return user_data