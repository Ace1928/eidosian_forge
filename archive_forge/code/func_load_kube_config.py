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
def load_kube_config(config_file=None, context=None, client_configuration=None, persist_config=True):
    """Loads authentication and cluster information from kube-config file

    and stores them in kubernetes.client.configuration.

    :param config_file: Name of the kube-config file.
    :param context: set the active context. If is set to None, current_context
        from config file will be used.
    :param client_configuration: The kubernetes.client.Configuration to
        set configs to.
    :param persist_config: If True, config file will be updated when changed
        (e.g GCP token refresh).
    """
    if config_file is None:
        config_file = KUBE_CONFIG_DEFAULT_LOCATION
    loader = _get_kube_config_loader_for_yaml_file(config_file, active_context=context, persist_config=persist_config)
    if client_configuration is None:
        config = type.__call__(Configuration)
        loader.load_and_set(config)
        Configuration.set_default(config)
    else:
        loader.load_and_set(client_configuration)