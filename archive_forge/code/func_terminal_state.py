import collections
import email.generator as generator
import email.mime.multipart as mime_multipart
import email.mime.nonmultipart as mime_nonmultipart
import email.parser as email_parser
import itertools
import time
import uuid
import six
from six.moves import http_client
from six.moves import urllib_parse
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
@property
def terminal_state(self):
    if self.__http_response is None:
        return False
    response_code = self.__http_response.status_code
    return response_code not in self.__retryable_codes