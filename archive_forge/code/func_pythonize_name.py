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
def pythonize_name(name):
    """Convert camel case to a "pythonic" name.

    Examples::

        pythonize_name('CamelCase') -> 'camel_case'
        pythonize_name('already_pythonized') -> 'already_pythonized'
        pythonize_name('HTTPRequest') -> 'http_request'
        pythonize_name('HTTPStatus200Ok') -> 'http_status_200_ok'
        pythonize_name('UPPER') -> 'upper'
        pythonize_name('') -> ''

    """
    s1 = _first_cap_regex.sub('\\1_\\2', name)
    s2 = _number_cap_regex.sub('\\1_\\2', s1)
    return _end_cap_regex.sub('\\1_\\2', s2).lower()