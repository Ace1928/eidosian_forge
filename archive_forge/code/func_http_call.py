from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def http_call(self, http_method, path, params=None, headers=None, data=None, files=None):
    """
        Execute an HTTP request.

        :param params: Dict of parameters to be sent in the request
        :param headers: Dict of headers to be sent in the request
        :param data: Binary data to be sent in the request
        :param files: Binary files to be sent in the request

        :return: :class:`dict` object
        :rtype: dict
        """
    full_path = urljoin(self.uri, path)
    kwargs = {'verify': self._session.verify}
    if headers:
        kwargs['headers'] = headers
    if params:
        if http_method in ['get', 'head']:
            kwargs['params'] = {k: _qs_param(v) for k, v in params.items()}
        else:
            kwargs['json'] = params
    elif http_method in ['post', 'put', 'patch'] and (not data) and (not files):
        kwargs['json'] = {}
    if files:
        kwargs['files'] = files
    if data:
        kwargs['data'] = data
    request = self._session.request(http_method, full_path, **kwargs)
    request.raise_for_status()
    self.validate_cache(request.headers.get('apipie-checksum'))
    if request.status_code == NO_CONTENT:
        return None
    return request.json()