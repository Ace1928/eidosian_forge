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
def get_with_name(self, name, safe=False):
    if not isinstance(self.value, list):
        raise ConfigException('Invalid kube-config file. Expected %s to be a list' % self.name)
    result = None
    for v in self.value:
        if 'name' not in v:
            raise ConfigException("Invalid kube-config file. Expected all values in %s list to have 'name' key" % self.name)
        if v['name'] == name:
            if result is None:
                result = v
            else:
                raise ConfigException('Invalid kube-config file. Expected only one object with name %s in %s list' % (name, self.name))
    if result is not None:
        if isinstance(result, ConfigNode):
            return result
        else:
            return ConfigNode('%s[name=%s]' % (self.name, name), result, self.path)
    if safe:
        return None
    raise ConfigException('Invalid kube-config file. Expected object with name %s in %s list' % (name, self.name))