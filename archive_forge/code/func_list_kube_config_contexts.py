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
def list_kube_config_contexts(config_file=None):
    if config_file is None:
        config_file = KUBE_CONFIG_DEFAULT_LOCATION
    loader = _get_kube_config_loader_for_yaml_file(config_file)
    return (loader.list_contexts(), loader.current_context)