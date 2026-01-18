from __future__ import absolute_import
import six
from six.moves import zip
from six import BytesIO
from six.moves import http_client
from six.moves.urllib.parse import urlencode, urlparse, urljoin, urlunparse, parse_qsl
import copy
from collections import OrderedDict
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
import json
import keyword
import logging
import mimetypes
import os
import re
import httplib2
import uritemplate
import google.api_core.client_options
from google.auth.transport import mtls
from google.auth.exceptions import MutualTLSChannelError
from googleapiclient import _auth
from googleapiclient import mimeparse
from googleapiclient.errors import HttpError
from googleapiclient.errors import InvalidJsonError
from googleapiclient.errors import MediaUploadSizeError
from googleapiclient.errors import UnacceptableMimeTypeError
from googleapiclient.errors import UnknownApiNameOrVersion
from googleapiclient.errors import UnknownFileType
from googleapiclient.http import build_http
from googleapiclient.http import BatchHttpRequest
from googleapiclient.http import HttpMock
from googleapiclient.http import HttpMockSequence
from googleapiclient.http import HttpRequest
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaUpload
from googleapiclient.model import JsonModel
from googleapiclient.model import MediaModel
from googleapiclient.model import RawModel
from googleapiclient.schema import Schemas
from googleapiclient._helpers import _add_query_parameter
from googleapiclient._helpers import positional
def methodNext(self, previous_request, previous_response):
    """Retrieves the next page of results.

Args:
  previous_request: The request for the previous page. (required)
  previous_response: The response from the request for the previous page. (required)

Returns:
  A request object that you can call 'execute()' on to request the next
  page. Returns None if there are no more items in the collection.
    """
    nextPageToken = previous_response.get(nextPageTokenName, None)
    if not nextPageToken:
        return None
    request = copy.copy(previous_request)
    if isPageTokenParameter:
        request.uri = _add_query_parameter(request.uri, pageTokenName, nextPageToken)
        logger.debug('Next page request URL: %s %s' % (methodName, request.uri))
    else:
        model = self._model
        body = model.deserialize(request.body)
        body[pageTokenName] = nextPageToken
        request.body = model.serialize(body)
        logger.debug('Next page request body: %s %s' % (methodName, body))
    return request