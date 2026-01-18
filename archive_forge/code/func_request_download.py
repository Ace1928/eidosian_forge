from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def request_download(self, path, destination=None, method='GET', api_version='v1'):
    if self.platform != 'nd':
        self.url = urljoin(self.baseuri, path)
    redirected = False
    redir_info = {}
    redirect = {}
    content = None
    data = None
    src = self.params.get('src')
    if src:
        try:
            self.headers.update({'Content-Length': os.stat(src).st_size})
            data = open(src, 'rb')
        except OSError:
            self.fail_json(msg='Unable to open source file %s' % src, elapsed=0)
    kwargs = {}
    if destination is not None and os.path.isdir(destination):
        if self.platform == 'nd':
            redir_info = self.connection.get_remote_file_io_stream(NDO_API_VERSION_PATH_FORMAT.format(api_version=api_version, path=path), self.module.tmpdir, method)
            content_disposition = redir_info.get('content-disposition')
        else:
            check, redir_info = fetch_url(self.module, self.url, headers=self.headers, method=method, timeout=self.params.get('timeout'))
            content_disposition = check.headers.get('Content-Disposition')
        if content_disposition:
            file_name = content_disposition.split('filename=')[1]
        else:
            self.fail_json(msg='Failed to fetch {0} backup information from MSO/NDO, response: {1}'.format(self.params.get('backup'), redir_info))
        if redir_info['status'] in (301, 302, 303, 307):
            self.url = redir_info.get('location')
            redirected = True
        destination = os.path.join(destination, file_name)
    if os.path.exists(destination):
        kwargs['last_mod_time'] = datetime.datetime.utcfromtimestamp(os.path.getmtime(destination))
    if self.platform == 'nd':
        if redir_info['status'] == 200 and redirected is False:
            info = redir_info
        else:
            info = self.connection.get_remote_file_io_stream('/mso/{0}'.format(self.url.split('/mso/', 1)), self.module.tmpdir, method)
    else:
        resp, info = fetch_url(self.module, self.url, data=data, headers=self.headers, method=method, timeout=self.params.get('timeout'), unix_socket=self.params.get('unix_socket'), **kwargs)
        try:
            content = resp.read()
        except AttributeError:
            content = info.pop('body', '')
        if src:
            try:
                data.close()
            except Exception:
                pass
    redirect['redirected'] = redirected or info.get('url') != self.url
    redirect.update(redir_info)
    redirect.update(info)
    write_file(self.module, self.url, destination, content, redirect, info.get('tmpsrc'))
    return (redirect, destination)