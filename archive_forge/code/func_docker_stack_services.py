from __future__ import (absolute_import, division, print_function)
import json
import os
import tempfile
import traceback
from ansible.module_utils.six import string_types
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
def docker_stack_services(client, stack_name):
    rc, out, err = client.call_cli('stack', 'services', stack_name, '--format', '{{.Name}}')
    if to_native(err) == 'Nothing found in stack: %s\n' % stack_name:
        return []
    return to_native(out).strip().split('\n')