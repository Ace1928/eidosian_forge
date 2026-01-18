from __future__ import (absolute_import, division, print_function)
import base64
import datetime
import io
import json
import os
import os.path
import shutil
import stat
import tarfile
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, NotFound
def fetch_file_ex(client, container, in_path, process_none, process_regular, process_symlink, process_other, follow_links=False, log=None):
    """Fetch a file (as a tar file entry) from a Docker container to local."""
    considered_in_paths = set()
    while True:
        if in_path in considered_in_paths:
            raise DockerFileCopyError('Found infinite symbolic link loop when trying to fetch "{0}"'.format(in_path))
        considered_in_paths.add(in_path)
        if log:
            log('FETCH: Fetching "%s"' % in_path)
        try:
            stream = client.get_raw_stream('/containers/{0}/archive', container, params={'path': in_path}, headers={'Accept-Encoding': 'identity'})
        except NotFound:
            return process_none(in_path)
        with tarfile.open(fileobj=_stream_generator_to_fileobj(stream), mode='r|') as tar:
            symlink_member = None
            result = None
            found = False
            for member in tar:
                if found:
                    raise DockerUnexpectedError('Received tarfile contains more than one file!')
                found = True
                if member.issym():
                    symlink_member = member
                    continue
                if member.isfile():
                    result = process_regular(in_path, tar, member)
                    continue
                result = process_other(in_path, member)
            if symlink_member:
                if not follow_links:
                    return process_symlink(in_path, symlink_member)
                in_path = os.path.join(os.path.split(in_path)[0], symlink_member.linkname)
                if log:
                    log('FETCH: Following symbolic link to "%s"' % in_path)
                continue
            if found:
                return result
            raise DockerUnexpectedError('Received tarfile is empty!')