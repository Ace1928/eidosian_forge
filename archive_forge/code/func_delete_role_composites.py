from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def delete_role_composites(self, rolerep, composites, clientid=None, realm='master'):
    composite_url = ''
    try:
        if clientid is not None:
            client = self.get_client_by_clientid(client_id=clientid, realm=realm)
            cid = client['id']
            composite_url = URL_CLIENT_ROLE_COMPOSITES.format(url=self.baseurl, realm=realm, id=cid, name=quote(rolerep['name'], safe=''))
        else:
            composite_url = URL_REALM_ROLE_COMPOSITES.format(url=self.baseurl, realm=realm, name=quote(rolerep['name'], safe=''))
        return open_url(composite_url, method='DELETE', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(composites), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not create role %s composites in realm %s: %s' % (rolerep['name'], realm, str(e)))