import datetime
import getpass
import json
import logging
import os
import subprocess
import threading
import time
from collections import namedtuple
from copy import deepcopy
from hashlib import sha1
from dateutil.parser import parse
from dateutil.tz import tzlocal, tzutc
import botocore.compat
import botocore.configloader
from botocore import UNSIGNED
from botocore.compat import compat_shell_split, total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.tokens import SSOTokenProvider
from botocore.utils import (
def fetch_credentials(require_expiry=True):
    credentials = {}
    access_key = environ.get(mapping['access_key'], '')
    if not access_key:
        raise PartialCredentialsError(provider=method, cred_var=mapping['access_key'])
    credentials['access_key'] = access_key
    secret_key = environ.get(mapping['secret_key'], '')
    if not secret_key:
        raise PartialCredentialsError(provider=method, cred_var=mapping['secret_key'])
    credentials['secret_key'] = secret_key
    credentials['token'] = None
    for token_env_var in mapping['token']:
        token = environ.get(token_env_var, '')
        if token:
            credentials['token'] = token
            break
    credentials['expiry_time'] = None
    expiry_time = environ.get(mapping['expiry_time'], '')
    if expiry_time:
        credentials['expiry_time'] = expiry_time
    if require_expiry and (not expiry_time):
        raise PartialCredentialsError(provider=method, cred_var=mapping['expiry_time'])
    return credentials