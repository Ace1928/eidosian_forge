from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_identity_providers(self, realm='master'):
    """ Fetch representations for identity providers in a realm
        :param realm: realm to be queried
        :return: list of representations for identity providers
        """
    idps_url = URL_IDENTITY_PROVIDERS.format(url=self.baseurl, realm=realm)
    try:
        return json.loads(to_native(open_url(idps_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except ValueError as e:
        self.module.fail_json(msg='API returned incorrect JSON when trying to obtain list of identity providers for realm %s: %s' % (realm, str(e)))
    except Exception as e:
        self.fail_open_url(e, msg='Could not obtain list of identity providers for realm %s: %s' % (realm, str(e)))