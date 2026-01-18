from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def lookup_remote_location(self, remote_location, ignore_not_found_error=False):
    """Look up a remote location and return its path and id"""
    if remote_location is None:
        return None
    remote = self.get_obj('platform/remote-locations', key='remoteLocations', name=remote_location)
    if 'id' not in remote and (not ignore_not_found_error):
        self.fail_json(msg="No remote location found for remote '{0}'".format(remote_location))
    elif 'id' not in remote and ignore_not_found_error:
        self.module.warn("No remote location found for remote '{0}'".format(remote_location))
        return dict()
    remote_info = dict(id=remote.get('id'), path=remote.get('credential')['remotePath'])
    return remote_info