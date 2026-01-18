import glob
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import pkgutil
import re
import urllib
from urllib import parse as urlparse
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1.identity import base
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
import requests
from cinderclient._i18n import _
from cinderclient import api_versions
from cinderclient import exceptions
import cinderclient.extension
def get_highest_client_server_version(url, insecure=False, cacert=None, cert=None):
    """Returns highest supported version by client and server as a string.

    :raises: UnsupportedVersion if the maximum supported by the server
             is less than the minimum supported by the client
    """
    min_server, max_server = get_server_version(url, insecure, cacert, cert)
    max_client = api_versions.APIVersion(api_versions.MAX_VERSION)
    min_client = api_versions.APIVersion(api_versions.MIN_VERSION)
    if max_server < min_client:
        msg = _('The maximum version supported by the server (%(srv)s) does not meet the minimum version supported by this client (%(cli)s)') % {'srv': str(max_server), 'cli': api_versions.MIN_VERSION}
        raise exceptions.UnsupportedVersion(msg)
    return min(max_server, max_client).get_string()