from __future__ import absolute_import
import six
from six.moves import http_client
from six.moves import range
from six import BytesIO, StringIO
from six.moves.urllib.parse import urlparse, urlunparse, quote, unquote
import copy
import httplib2
import json
import logging
import mimetypes
import os
import random
import socket
import time
import uuid
from email.generator import Generator
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
from email.parser import FeedParser
from googleapiclient import _helpers as util
from googleapiclient import _auth
from googleapiclient.errors import BatchError
from googleapiclient.errors import HttpError
from googleapiclient.errors import InvalidChunkSizeError
from googleapiclient.errors import ResumableUploadError
from googleapiclient.errors import UnexpectedBodyError
from googleapiclient.errors import UnexpectedMethodError
from googleapiclient.model import JsonModel
def set_user_agent(http, user_agent):
    """Set the user-agent on every request.

  Args:
     http - An instance of httplib2.Http
         or something that acts like it.
     user_agent: string, the value for the user-agent header.

  Returns:
     A modified instance of http that was passed in.

  Example:

    h = httplib2.Http()
    h = set_user_agent(h, "my-app-name/6.0")

  Most of the time the user-agent will be set doing auth, this is for the rare
  cases where you are accessing an unauthenticated endpoint.
  """
    request_orig = http.request

    def new_request(uri, method='GET', body=None, headers=None, redirections=httplib2.DEFAULT_MAX_REDIRECTS, connection_type=None):
        """Modify the request headers to add the user-agent."""
        if headers is None:
            headers = {}
        if 'user-agent' in headers:
            headers['user-agent'] = user_agent + ' ' + headers['user-agent']
        else:
            headers['user-agent'] = user_agent
        resp, content = request_orig(uri, method=method, body=body, headers=headers, redirections=redirections, connection_type=connection_type)
        return (resp, content)
    http.request = new_request
    return http