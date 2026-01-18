import copy
import functools
import getpass
import logging
import os
import time
import warnings
from cliff import columns as cliff_columns
from oslo_utils import importutils
from osc_lib import exceptions
from osc_lib.i18n import _
def get_client_class(api_name, version, version_map):
    """Returns the client class for the requested API version

    :param api_name: the name of the API, e.g. 'compute', 'image', etc
    :param version: the requested API version
    :param version_map: a dict of client classes keyed by version
    :rtype: a client class for the requested API version
    """
    try:
        client_path = version_map[str(version)]
    except (KeyError, ValueError):
        sorted_versions = sorted(version_map.keys(), key=lambda s: list(map(int, s.split('.'))))
        msg = _("Invalid %(api_name)s client version '%(version)s'. must be one of: %(version_map)s")
        raise exceptions.UnsupportedVersion(msg % {'api_name': api_name, 'version': version, 'version_map': ', '.join(sorted_versions)})
    return importutils.import_class(client_path)