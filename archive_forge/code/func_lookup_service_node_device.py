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
def lookup_service_node_device(self, site_id, tenant, device_name=None, service_node_type=None, ignore_not_found_error=False):
    if service_node_type is None:
        node_devices = self.query_objs('sites/{0}/aci/tenants/{1}/devices'.format(site_id, tenant), key='devices')
    else:
        node_devices = self.query_objs('sites/{0}/aci/tenants/{1}/devices?deviceType={2}'.format(site_id, tenant, service_node_type), key='devices')
    if device_name is not None:
        for device in node_devices:
            if device_name == device.get('name'):
                return device
        if ignore_not_found_error:
            self.module.warn("Provided device '{0}' of type '{1}' does not exist.".format(device_name, service_node_type))
            return node_devices
        else:
            self.module.fail_json(msg="Provided device '{0}' of type '{1}' does not exist.".format(device_name, service_node_type))
    return node_devices