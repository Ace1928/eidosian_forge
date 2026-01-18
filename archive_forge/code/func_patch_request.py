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
def patch_request(self, uri, pyld, check_pyld=False):
    req_headers = dict(PATCH_HEADERS)
    r = self.get_request(uri)
    if r['ret']:
        etag = r['headers'].get('etag')
        if not etag:
            etag = r['data'].get('@odata.etag')
        if etag:
            if self.strip_etag_quotes:
                etag = etag.strip('"')
            req_headers['If-Match'] = etag
    if check_pyld:
        if r['ret']:
            check_resp = self._check_request_payload(pyld, r['data'], uri)
            if not check_resp.pop('changes_required'):
                check_resp['changed'] = False
                return check_resp
        else:
            r['changed'] = False
            return r
    username, password, basic_auth = self._auth_params(req_headers)
    try:
        resp = open_url(uri, data=json.dumps(pyld), headers=req_headers, method='PATCH', url_username=username, url_password=password, force_basic_auth=basic_auth, validate_certs=False, follow_redirects='all', use_proxy=True, timeout=self.timeout)
    except HTTPError as e:
        msg = self._get_extended_message(e)
        return {'ret': False, 'changed': False, 'msg': "HTTP Error %s on PATCH request to '%s', extended message: '%s'" % (e.code, uri, msg), 'status': e.code}
    except URLError as e:
        return {'ret': False, 'changed': False, 'msg': "URL Error on PATCH request to '%s': '%s'" % (uri, e.reason)}
    except Exception as e:
        return {'ret': False, 'changed': False, 'msg': "Failed PATCH request to '%s': '%s'" % (uri, to_text(e))}
    return {'ret': True, 'changed': True, 'resp': resp, 'msg': 'Modified %s' % uri}