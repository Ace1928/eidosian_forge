from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def put_request(self, uri, pyld):
    req_headers = dict(PUT_HEADERS)
    r = self.get_request(uri)
    if r['ret']:
        etag = r['headers'].get('etag')
        if not etag:
            etag = r['data'].get('@odata.etag')
        if etag:
            if self.strip_etag_quotes:
                etag = etag.strip('"')
            req_headers['If-Match'] = etag
    username, password, basic_auth = self._auth_params(req_headers)
    try:
        resp = open_url(uri, data=json.dumps(pyld), headers=req_headers, method='PUT', url_username=username, url_password=password, force_basic_auth=basic_auth, validate_certs=False, follow_redirects='all', use_proxy=True, timeout=self.timeout)
    except HTTPError as e:
        msg = self._get_extended_message(e)
        return {'ret': False, 'msg': "HTTP Error %s on PUT request to '%s', extended message: '%s'" % (e.code, uri, msg), 'status': e.code}
    except URLError as e:
        return {'ret': False, 'msg': "URL Error on PUT request to '%s': '%s'" % (uri, e.reason)}
    except Exception as e:
        return {'ret': False, 'msg': "Failed PUT request to '%s': '%s'" % (uri, to_text(e))}
    return {'ret': True, 'resp': resp}