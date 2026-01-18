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
def providers(self, profile_name, disable_env_vars=False):
    return [self._create_web_identity_provider(profile_name, disable_env_vars), self._create_sso_provider(profile_name), self._create_shared_credential_provider(profile_name), self._create_process_provider(profile_name), self._create_config_provider(profile_name)]