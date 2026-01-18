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
def source_credentials(self, source_name):
    """Loads source credentials based on the provided configuration.

        :type source_name: str
        :param source_name: The value of credential_source in the config
            file. This is the canonical name of the credential provider.

        :rtype: Credentials
        """
    source = self._get_provider(source_name)
    if isinstance(source, CredentialResolver):
        return source.load_credentials()
    return source.load()