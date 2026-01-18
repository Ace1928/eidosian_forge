from __future__ import (absolute_import, division, print_function)
import json
import os
import tempfile
import traceback
from ansible.module_utils.six import string_types
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
def docker_stack_inspect(client, stack_name):
    ret = {}
    for service_name in docker_stack_services(client, stack_name):
        ret[service_name] = docker_service_inspect(client, service_name)
    return ret