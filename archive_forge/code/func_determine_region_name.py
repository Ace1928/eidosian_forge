import base64
import boto
import boto.auth_handler
import boto.exception
import boto.plugin
import boto.utils
import copy
import datetime
from email.utils import formatdate
import hmac
import os
import posixpath
from boto.compat import urllib, encodebytes, parse_qs_safe, urlparse, six
from boto.auth_handler import AuthHandler
from boto.exception import BotoClientError
from boto.utils import get_utf8able_str
def determine_region_name(self, host):
    parts = self.split_host_parts(host)
    if self.region_name is not None:
        region_name = self.region_name
    elif len(parts) == 3:
        region_name = self.clean_region_name(parts[0])
        if region_name == 's3':
            region_name = 'us-east-1'
    else:
        for offset, part in enumerate(reversed(parts)):
            part = part.lower()
            if part == 's3':
                region_name = parts[-offset]
                if region_name == 'amazonaws':
                    region_name = 'us-east-1'
                break
            elif part.startswith('s3-'):
                region_name = self.clean_region_name(part)
                break
    return region_name