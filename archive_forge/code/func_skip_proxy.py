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
def skip_proxy(self, host):
    if not self.no_proxy:
        return False
    if self.no_proxy == '*':
        return True
    hostonly = host
    hostonly = host.split(':')[0]
    for name in self.no_proxy.split(','):
        if name and (hostonly.endswith(name) or host.endswith(name)):
            return True
    return False