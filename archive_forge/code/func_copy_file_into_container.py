from __future__ import absolute_import, division, print_function
import base64
import io
import os
import stat
import traceback
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, DockerException, NotFound
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.copy import (
from ansible_collections.community.docker.plugins.module_utils._scramble import generate_insecure_key, scramble
def copy_file_into_container(client, container, managed_path, container_path, follow_links, local_follow_links, owner_id, group_id, mode, force=False, diff=False, max_file_size_for_diff=1):
    if diff:
        diff = {}
    else:
        diff = None
    container_path, mode, idempotent = is_file_idempotent(client, container, managed_path, container_path, follow_links, local_follow_links, owner_id, group_id, mode, force=force, diff=diff, max_file_size_for_diff=max_file_size_for_diff)
    changed = not idempotent
    if changed and (not client.module.check_mode):
        put_file(client, container, in_path=managed_path, out_path=container_path, user_id=owner_id, group_id=group_id, mode=mode, follow_links=local_follow_links)
    result = dict(container_path=container_path, changed=changed)
    if diff:
        result['diff'] = diff
    client.module.exit_json(**result)