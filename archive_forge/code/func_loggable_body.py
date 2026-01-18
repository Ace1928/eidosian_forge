import collections
import contextlib
import logging
import socket
import time
import httplib2
import six
from six.moves import http_client
from six.moves.urllib import parse
from apitools.base.py import exceptions
from apitools.base.py import util
@loggable_body.setter
def loggable_body(self, value):
    if self.body is None:
        raise exceptions.RequestError('Cannot set loggable body on request with no body')
    self.__loggable_body = value