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
def load_credentials(self):
    """
        Goes through the credentials chain, returning the first ``Credentials``
        that could be loaded.
        """
    for provider in self.providers:
        logger.debug('Looking for credentials via: %s', provider.METHOD)
        creds = provider.load()
        if creds is not None:
            return creds
    return None