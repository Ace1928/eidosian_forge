from datetime import datetime
import errno
import os
import random
import re
import socket
import sys
import time
import xml.sax
import copy
from boto import auth
from boto import auth_handler
import boto
import boto.utils
import boto.handler
import boto.cacerts
from boto import config, UserAgent
from boto.compat import six, http_client, urlparse, quote, encodebytes
from boto.exception import AWSConnectionError
from boto.exception import BotoClientError
from boto.exception import BotoServerError
from boto.exception import PleaseRetryException
from boto.exception import S3ResponseError
from boto.provider import Provider
from boto.resultset import ResultSet
def new_http_connection(self, host, port, is_secure):
    if host is None:
        host = self.server_name()
    host = boto.utils.parse_host(host)
    http_connection_kwargs = self.http_connection_kwargs.copy()
    http_connection_kwargs['port'] = port
    if self.use_proxy and (not is_secure) and (not self.skip_proxy(host)):
        host = self.proxy
        http_connection_kwargs['port'] = int(self.proxy_port)
    if is_secure:
        boto.log.debug('establishing HTTPS connection: host=%s, kwargs=%s', host, http_connection_kwargs)
        if self.use_proxy and (not self.skip_proxy(host)):
            connection = self.proxy_ssl(host, is_secure and 443 or 80)
        elif self.https_connection_factory:
            connection = self.https_connection_factory(host)
        elif self.https_validate_certificates and HAVE_HTTPS_CONNECTION:
            connection = https_connection.CertValidatingHTTPSConnection(host, ca_certs=self.ca_certificates_file, **http_connection_kwargs)
        else:
            connection = http_client.HTTPSConnection(host, **http_connection_kwargs)
    else:
        boto.log.debug('establishing HTTP connection: kwargs=%s' % http_connection_kwargs)
        if self.https_connection_factory:
            connection = self.https_connection_factory(host, **http_connection_kwargs)
        else:
            connection = http_client.HTTPConnection(host, **http_connection_kwargs)
    if self.debug > 1:
        connection.set_debuglevel(self.debug)
    if host.split(':')[0] == self.host and is_secure == self.is_secure:
        self._connection = (host, port, is_secure)
    connection.response_class = HTTPResponse
    return connection