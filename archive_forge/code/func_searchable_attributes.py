from __future__ import (absolute_import, division, print_function)
import json
import os
import subprocess
import time
import traceback
import inspect
from ansible.module_utils.six.moves.http_client import HTTPSConnection, IncompleteRead
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def searchable_attributes(module):
    """
    Return all searchable disk attributes passed to module.
    """
    attributes = {'name': module.params.get('name'), 'Storage.name': module.params.get('storage_domain'), 'vm_names': module.params.get('vm_name') if module.params.get('state') != 'attached' else None}
    return dict(((k, v) for k, v in attributes.items() if v is not None))