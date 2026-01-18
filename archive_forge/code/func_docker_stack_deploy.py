from __future__ import (absolute_import, division, print_function)
import json
import os
import tempfile
import traceback
from ansible.module_utils.six import string_types
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
def docker_stack_deploy(client, stack_name, compose_files):
    command = ['stack', 'deploy']
    if client.module.params['prune']:
        command += ['--prune']
    if client.module.params['with_registry_auth']:
        command += ['--with-registry-auth']
    if client.module.params['resolve_image']:
        command += ['--resolve-image', client.module.params['resolve_image']]
    for compose_file in compose_files:
        command += ['--compose-file', compose_file]
    command += [stack_name]
    rc, out, err = client.call_cli(*command)
    return (rc, to_native(out), to_native(err))