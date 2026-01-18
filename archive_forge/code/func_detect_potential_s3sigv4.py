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
def detect_potential_s3sigv4(func):

    def _wrapper(self):
        env_use_sigv4_flag = os.environ.get('S3_USE_SIGV4')
        cfg_use_sigv4_flag = boto.config.get('s3', 'use-sigv4')
        for flag in (env_use_sigv4_flag, cfg_use_sigv4_flag):
            flag = convert_to_bool(flag)
            if flag is not None:
                return ['hmac-v4-s3'] if flag else func(self)
        host = self.host
        if not self.host.startswith('http://') or self.host.startswith('https://'):
            host = 'https://' + host
        netloc = urlparse(host).netloc
        if not (netloc.endswith('amazonaws.com') or netloc.endswith('amazonaws.com.cn')):
            return func(self)
        if hasattr(self, 'anon') and self.anon:
            return func(self)
        return ['hmac-v4-s3']
    return _wrapper