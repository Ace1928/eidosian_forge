from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_component(self, cid, realm='master'):
    """ Fetch component representation from a realm using its cid.
        If the component does not exist, None is returned.
        :param cid: Unique ID of the component to fetch.
        :param realm: Realm in which the component resides; default 'master'.
        """
    comp_url = URL_COMPONENT.format(url=self.baseurl, realm=realm, id=cid)
    try:
        return json.loads(to_native(open_url(comp_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except HTTPError as e:
        if e.code == 404:
            return None
        else:
            self.fail_open_url(e, msg='Could not fetch component %s in realm %s: %s' % (cid, realm, str(e)))
    except Exception as e:
        self.module.fail_json(msg='Could not fetch component %s in realm %s: %s' % (cid, realm, str(e)))