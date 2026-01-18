import base64
import calendar
import datetime
import functools
import hmac
import json
import logging
import time
from collections.abc import Mapping
from email.utils import formatdate
from hashlib import sha1, sha256
from operator import itemgetter
from botocore.compat import (
from botocore.exceptions import NoAuthTokenError, NoCredentialsError
from botocore.utils import (
from botocore.compat import MD5_AVAILABLE  # noqa
def canonical_resource(self, split, auth_path=None):
    if auth_path is not None:
        buf = auth_path
    else:
        buf = split.path
    if split.query:
        qsa = split.query.split('&')
        qsa = [a.split('=', 1) for a in qsa]
        qsa = [self.unquote_v(a) for a in qsa if a[0] in self.QSAOfInterest]
        if len(qsa) > 0:
            qsa.sort(key=itemgetter(0))
            qsa = ['='.join(a) for a in qsa]
            buf += '?'
            buf += '&'.join(qsa)
    return buf