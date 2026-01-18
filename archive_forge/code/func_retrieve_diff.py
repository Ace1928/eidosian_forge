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
def retrieve_diff(client, container, container_path, follow_links, diff, max_file_size_for_diff, regular_stat=None, link_target=None):
    if diff is None:
        return
    if regular_stat is not None:
        if regular_stat['mode'] & 1 << 32 - 1 != 0:
            diff['before_header'] = container_path
            diff['before'] = '(directory)'
            return
        elif regular_stat['mode'] & 1 << 32 - 4 != 0:
            diff['before_header'] = container_path
            diff['before'] = '(temporary file)'
            return
        elif regular_stat['mode'] & 1 << 32 - 5 != 0:
            diff['before_header'] = container_path
            diff['before'] = link_target
            return
        elif regular_stat['mode'] & 1 << 32 - 6 != 0:
            diff['before_header'] = container_path
            diff['before'] = '(device)'
            return
        elif regular_stat['mode'] & 1 << 32 - 7 != 0:
            diff['before_header'] = container_path
            diff['before'] = '(named pipe)'
            return
        elif regular_stat['mode'] & 1 << 32 - 8 != 0:
            diff['before_header'] = container_path
            diff['before'] = '(socket)'
            return
        elif regular_stat['mode'] & 1 << 32 - 11 != 0:
            diff['before_header'] = container_path
            diff['before'] = '(character device)'
            return
        elif regular_stat['mode'] & 1 << 32 - 13 != 0:
            diff['before_header'] = container_path
            diff['before'] = '(unknown filesystem object)'
            return
        if regular_stat['size'] > max_file_size_for_diff > 0:
            diff['dst_larger'] = max_file_size_for_diff
            return

    def process_none(in_path):
        diff['before'] = ''

    def process_regular(in_path, tar, member):
        add_diff_dst_from_regular_member(diff, max_file_size_for_diff, in_path, tar, member)

    def process_symlink(in_path, member):
        diff['before_header'] = in_path
        diff['before'] = member.linkname

    def process_other(in_path, member):
        add_other_diff(diff, in_path, member)
    fetch_file_ex(client, container, in_path=container_path, process_none=process_none, process_regular=process_regular, process_symlink=process_symlink, process_other=process_other, follow_links=follow_links)