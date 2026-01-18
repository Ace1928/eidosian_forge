from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_apps_dict(array):
    apps_info = {}
    api_version = array._list_available_rest_versions()
    if SAN_REQUIRED_API_VERSION in api_version:
        apps = array.list_apps()
        for app in range(0, len(apps)):
            appname = apps[app]['name']
            apps_info[appname] = {'version': apps[app]['version'], 'status': apps[app]['status'], 'description': apps[app]['description']}
    if P53_API_VERSION in api_version:
        app_nodes = array.list_app_nodes()
        for app in range(0, len(app_nodes)):
            appname = app_nodes[app]['name']
            apps_info[appname]['index'] = app_nodes[app]['index']
            apps_info[appname]['vnc'] = app_nodes[app]['vnc']
    return apps_info