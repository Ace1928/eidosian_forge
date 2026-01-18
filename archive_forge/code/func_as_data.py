import atexit
import base64
import copy
import datetime
import json
import logging
import os
import platform
import tempfile
import time
import google.auth
import google.auth.transport.requests
import oauthlib.oauth2
import urllib3
from ruamel import yaml
from requests_oauthlib import OAuth2Session
from six import PY3
from kubernetes.client import ApiClient, Configuration
from kubernetes.config.exec_provider import ExecProvider
from .config_exception import ConfigException
from .dateutil import UTC, format_rfc3339, parse_rfc3339
def as_data(self):
    """If obj[%data_key_name] exists, Return obj[%data_key_name] otherwise

        base64 encoded string of obj[%file_key_name] file content.
    """
    use_file_if_no_data = not self._data and self._file
    if use_file_if_no_data:
        with open(self._file) as f:
            if self._base64_file_content:
                self._data = bytes.decode(base64.standard_b64encode(str.encode(f.read())))
            else:
                self._data = f.read()
    return self._data