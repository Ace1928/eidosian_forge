from __future__ import (absolute_import, division, print_function)
import json
import os
import tempfile
import traceback
from ansible.module_utils.six import string_types
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
def docker_service_inspect(client, service_name):
    rc, out, err = client.call_cli('service', 'inspect', service_name)
    if rc != 0:
        return None
    else:
        ret = json.loads(out)[0]['Spec']
        return ret