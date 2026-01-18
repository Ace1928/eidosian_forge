import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
def make_size_human_readable(size):
    suffix = ['B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB']
    base = 1024.0
    index = 0
    if size is None:
        size = 0
    while size >= base:
        index = index + 1
        size = size / base
    padded = '%.1f' % size
    stripped = padded.rstrip('0').rstrip('.')
    return '%s%s' % (stripped, suffix[index])