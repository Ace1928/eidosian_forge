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
def set_active_context(self, context_name=None):
    if context_name is None:
        context_name = self._config['current-context']
    self._current_context = self._config['contexts'].get_with_name(context_name)
    if self._current_context['context'].safe_get('user') and self._config.safe_get('users'):
        user = self._config['users'].get_with_name(self._current_context['context']['user'], safe=True)
        if user:
            self._user = user['user']
        else:
            self._user = None
    else:
        self._user = None
    self._cluster = self._config['clusters'].get_with_name(self._current_context['context']['cluster'])['cluster']