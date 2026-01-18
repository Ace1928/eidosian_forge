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
def guess_mime_type(content, deftype):
    """Description: Guess the mime type of a block of text
    :param content: content we're finding the type of
    :type str:

    :param deftype: Default mime type
    :type str:

    :rtype: <type>:
    :return: <description>
    """
    starts_with_mappings = {'#include': 'text/x-include-url', '#!': 'text/x-shellscript', '#cloud-config': 'text/cloud-config', '#upstart-job': 'text/upstart-job', '#part-handler': 'text/part-handler', '#cloud-boothook': 'text/cloud-boothook'}
    rtype = deftype
    for possible_type, mimetype in starts_with_mappings.items():
        if content.startswith(possible_type):
            rtype = mimetype
            break
    return rtype