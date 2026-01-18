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
def print_to_fd(*objects, **kwargs):
    """A Python 2/3 compatible analogue to the print function.

    This function writes text to a file descriptor as the
    builtin print function would, favoring utf-8 encoding.
    Arguments and return values are the same as documented in
    the Python 2 print function.
    """

    def _get_args(**kwargs):
        """Validates keyword arguments that would be used in Print
        Valid keyword arguments, mirroring print(), are 'sep',
        'end', and 'file'. These must be of types string, string,
        and file / file interface respectively.
        Returns the above kwargs of the above types.
        """
        expected_keywords = collections.OrderedDict([('sep', ' '), ('end', '\n'), ('file', sys.stdout)])
        for key, value in kwargs.items():
            if key not in expected_keywords:
                error_msg = '{} is not a valid keyword argument. Please use one of: {}'
                raise KeyError(error_msg.format(key, ' '.join(expected_keywords.keys())))
            else:
                expected_keywords[key] = value
        return expected_keywords.values()

    def _get_byte_strings(*objects):
        """Gets a `bytes` string for each item in list of printable objects."""
        byte_objects = []
        for item in objects:
            if not isinstance(item, (six.binary_type, six.text_type)):
                item = str(item)
            if isinstance(item, six.binary_type):
                byte_objects.append(item)
            else:
                byte_objects.append(six.ensure_binary(item))
        return byte_objects
    sep, end, file = _get_args(**kwargs)
    sep = six.ensure_binary(sep)
    end = six.ensure_binary(end)
    data = _get_byte_strings(*objects)
    data = sep.join(data)
    data += end
    write_to_fd(file, data)