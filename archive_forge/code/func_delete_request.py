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
def delete_request(self, uri, pyld=None):
    req_headers = dict(DELETE_HEADERS)
    username, password, basic_auth = self._auth_params(req_headers)
    try:
        data = json.dumps(pyld) if pyld else None
        resp = open_url(uri, data=data, headers=req_headers, method='DELETE', url_username=username, url_password=password, force_basic_auth=basic_auth, validate_certs=False, follow_redirects='all', use_proxy=True, timeout=self.timeout)
    except HTTPError as e:
        msg = self._get_extended_message(e)
        return {'ret': False, 'msg': "HTTP Error %s on DELETE request to '%s', extended message: '%s'" % (e.code, uri, msg), 'status': e.code}
    except URLError as e:
        return {'ret': False, 'msg': "URL Error on DELETE request to '%s': '%s'" % (uri, e.reason)}
    except Exception as e:
        return {'ret': False, 'msg': "Failed DELETE request to '%s': '%s'" % (uri, to_text(e))}
    return {'ret': True, 'resp': resp}