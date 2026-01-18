from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_client_by_id(self, id, realm='master'):
    """ Obtain client representation by id

        :param id: id (not clientId) of client to be queried
        :param realm: client from this realm
        :return: dict of client representation or None if none matching exist
        """
    client_url = URL_CLIENT.format(url=self.baseurl, realm=realm, id=id)
    try:
        return json.loads(to_native(open_url(client_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except HTTPError as e:
        if e.code == 404:
            return None
        else:
            self.fail_open_url(e, msg='Could not obtain client %s for realm %s: %s' % (id, realm, str(e)))
    except ValueError as e:
        self.module.fail_json(msg='API returned incorrect JSON when trying to obtain client %s for realm %s: %s' % (id, realm, str(e)))
    except Exception as e:
        self.module.fail_json(msg='Could not obtain client %s for realm %s: %s' % (id, realm, str(e)))