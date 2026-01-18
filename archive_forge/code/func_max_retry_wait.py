import base64
import contextlib
import datetime
import logging
import pprint
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import util
@max_retry_wait.setter
def max_retry_wait(self, value):
    util.Typecheck(value, six.integer_types)
    if value <= 0:
        raise exceptions.InvalidDataError('max_retry_wait must be a postiive integer')
    self.__max_retry_wait = value