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
def is_content_idempotent(client, container, content, container_path, follow_links, owner_id, group_id, mode, force=False, diff=None, max_file_size_for_diff=1):
    if diff is not None:
        if len(content) > max_file_size_for_diff > 0:
            diff['src_larger'] = max_file_size_for_diff
        elif is_binary(content):
            diff['src_binary'] = 1
        else:
            diff['after_header'] = 'dynamically generated'
            diff['after'] = to_text(content)
    if force and (not follow_links):
        retrieve_diff(client, container, container_path, follow_links, diff, max_file_size_for_diff)
        return (container_path, mode, False)
    real_container_path, regular_stat, link_target = stat_file(client, container, in_path=container_path, follow_links=follow_links)
    if follow_links:
        container_path = real_container_path
    if regular_stat is None:
        if diff is not None:
            diff['before_header'] = container_path
            diff['before'] = ''
        return (container_path, mode, False)
    if force:
        retrieve_diff(client, container, container_path, follow_links, diff, max_file_size_for_diff, regular_stat, link_target)
        return (container_path, mode, False)
    if force is False:
        retrieve_diff(client, container, container_path, follow_links, diff, max_file_size_for_diff, regular_stat, link_target)
        copy_dst_to_src(diff)
        return (container_path, mode, True)
    if link_target is not None:
        retrieve_diff(client, container, container_path, follow_links, diff, max_file_size_for_diff, regular_stat, link_target)
        return (container_path, mode, False)
    if is_container_file_not_regular_file(regular_stat):
        retrieve_diff(client, container, container_path, follow_links, diff, max_file_size_for_diff, regular_stat, link_target)
        return (container_path, mode, False)
    if len(content) != regular_stat['size']:
        retrieve_diff(client, container, container_path, follow_links, diff, max_file_size_for_diff, regular_stat, link_target)
        return (container_path, mode, False)
    if mode != get_container_file_mode(regular_stat):
        retrieve_diff(client, container, container_path, follow_links, diff, max_file_size_for_diff, regular_stat, link_target)
        return (container_path, mode, False)

    def process_none(in_path):
        if diff is not None:
            diff['before'] = ''
        return (container_path, mode, False)

    def process_regular(in_path, tar, member):
        if any([member.mode & 4095 != mode, member.uid != owner_id, member.gid != group_id, member.size != len(content)]):
            add_diff_dst_from_regular_member(diff, max_file_size_for_diff, in_path, tar, member)
            return (container_path, mode, False)
        tar_f = tar.extractfile(member)
        is_equal = are_fileobjs_equal_with_diff_of_first(tar_f, io.BytesIO(content), member.size, diff, max_file_size_for_diff, in_path)
        return (container_path, mode, is_equal)

    def process_symlink(in_path, member):
        if diff is not None:
            diff['before_header'] = in_path
            diff['before'] = member.linkname
        return (container_path, mode, False)

    def process_other(in_path, member):
        add_other_diff(diff, in_path, member)
        return (container_path, mode, False)
    return fetch_file_ex(client, container, in_path=container_path, process_none=process_none, process_regular=process_regular, process_symlink=process_symlink, process_other=process_other, follow_links=follow_links)