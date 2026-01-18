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
def add_mtls_creds(http, client_options, adc_cert_path, adc_key_path):
    """Obtain client cert and create mTLS http channel if cert exists.
  Args:
    http: httplib2.Http, An instance of httplib2.Http or something that acts
      like it that HTTP requests will be made through.
    client_options: Mapping object or google.api_core.client_options, client
      options to set user options on the client.
    adc_cert_path: str, client certificate file path to save the application
      default client certificate for mTLS.
    adc_key_path: str, client encrypted private key file path to save the
      application default client encrypted private key for mTLS.
  Returns:
    Boolean indicating whether the cert was used or not.
  """
    client_cert_to_use = None
    if client_options and client_options.client_cert_source:
        raise MutualTLSChannelError('ClientOptions.client_cert_source is not supported, please use ClientOptions.client_encrypted_cert_source.')
    if client_options and hasattr(client_options, 'client_encrypted_cert_source') and client_options.client_encrypted_cert_source:
        client_cert_to_use = client_options.client_encrypted_cert_source
    elif adc_cert_path and adc_key_path and mtls.has_default_client_cert_source():
        client_cert_to_use = mtls.default_client_encrypted_cert_source(adc_cert_path, adc_key_path)
    if not client_cert_to_use:
        return False
    cert_path, key_path, passphrase = client_cert_to_use()
    http_channel = http.http if google_auth_httplib2 and isinstance(http, google_auth_httplib2.AuthorizedHttp) else http
    http_channel.add_certificate(key_path, cert_path, '', passphrase)
    return True