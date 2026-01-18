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
def create_assume_role_refresher(client, params):

    def refresh():
        response = client.assume_role(**params)
        credentials = response['Credentials']
        return {'access_key': credentials['AccessKeyId'], 'secret_key': credentials['SecretAccessKey'], 'token': credentials['SessionToken'], 'expiry_time': _serialize_if_needed(credentials['Expiration'])}
    return refresh