from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def machine_type_selflink(name, params):
    if name is None:
        return
    url = 'https://www.googleapis.com/compute/v1/projects/.*/zones/.*/machineTypes/.*'
    if not re.match(url, name):
        name = 'https://www.googleapis.com/compute/v1/projects/{project}/zones/{zone}/machineTypes/%s'.format(**params) % name
    return name