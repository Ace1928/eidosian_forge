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
def handle_proxy(self, proxy, proxy_port, proxy_user, proxy_pass):
    self.proxy = proxy
    self.proxy_port = proxy_port
    self.proxy_user = proxy_user
    self.proxy_pass = proxy_pass
    if 'http_proxy' in os.environ and (not self.proxy):
        pattern = re.compile('(?:http://)?(?:(?P<user>[\\w\\-\\.]+):(?P<pass>.*)@)?(?P<host>[\\w\\-\\.]+)(?::(?P<port>\\d+))?')
        match = pattern.match(os.environ['http_proxy'])
        if match:
            self.proxy = match.group('host')
            self.proxy_port = match.group('port')
            self.proxy_user = match.group('user')
            self.proxy_pass = match.group('pass')
    else:
        if not self.proxy:
            self.proxy = config.get_value('Boto', 'proxy', None)
        if not self.proxy_port:
            self.proxy_port = config.get_value('Boto', 'proxy_port', None)
        if not self.proxy_user:
            self.proxy_user = config.get_value('Boto', 'proxy_user', None)
        if not self.proxy_pass:
            self.proxy_pass = config.get_value('Boto', 'proxy_pass', None)
    if not self.proxy_port and self.proxy:
        print('http_proxy environment variable does not specify a port, using default')
        self.proxy_port = self.port
    self.no_proxy = os.environ.get('no_proxy', '') or os.environ.get('NO_PROXY', '')
    self.use_proxy = self.proxy is not None