from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.urls import urlparse
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.urls import Request
from .common import F5ModuleError
from ansible.module_utils.six.moves.urllib.error import HTTPError
from .constants import (
def download_asm_file(client, url, dest, file_size):
    """Download a large ASM file from the remote device

    This method handles issues with ASM file endpoints that allow
    downloads of ASM objects on the BIG-IP, as well as handles
    chunking of large files.

    Arguments:
        client (object): The F5RestClient connection object.
        url (string): The URL to download.
        dest (string): The location on (Ansible controller) disk to store the file.
        file_size (integer): The size of the remote file.

    Returns:
        bool: No response on success. Fail otherwise.
    """
    with open(dest, 'wb') as fileobj:
        chunk_size = 512 * 1024
        start = 0
        end = chunk_size - 1
        size = file_size
        while True:
            content_range = '%s-%s/%s' % (start, end, size)
            headers = {'Content-Range': content_range, 'Content-Type': 'application/json'}
            data = {'headers': headers, 'verify': False, 'stream': False}
            response = client.api.get(url, headers=headers, json=data)
            if response.status == 200:
                if 'Content-Length' not in response.headers:
                    error_message = 'The Content-Length header is not present.'
                    raise F5ModuleError(error_message)
                length = response.headers['Content-Length']
                if int(length) > 0:
                    fileobj.write(response.content)
                else:
                    error = 'Invalid Content-Length value returned: %s ,the value should be greater than 0' % length
                    raise F5ModuleError(error)
            if end == size:
                break
            start += chunk_size
            if start >= size:
                break
            if end + chunk_size > size:
                end = size - 1
            else:
                end = start + chunk_size - 1