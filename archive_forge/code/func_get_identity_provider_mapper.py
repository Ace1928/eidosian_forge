from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_identity_provider_mapper(self, mid, alias, realm='master'):
    """ Fetch identity provider representation from a realm using the idp's alias.
        If the identity provider does not exist, None is returned.
        :param mid: Unique ID of the mapper to fetch.
        :param alias: Alias of the identity provider.
        :param realm: Realm in which the identity provider resides; default 'master'.
        """
    mapper_url = URL_IDENTITY_PROVIDER_MAPPER.format(url=self.baseurl, realm=realm, alias=alias, id=mid)
    try:
        return json.loads(to_native(open_url(mapper_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except HTTPError as e:
        if e.code == 404:
            return None
        else:
            self.fail_open_url(e, msg='Could not fetch mapper %s for identity provider %s in realm %s: %s' % (mid, alias, realm, str(e)))
    except Exception as e:
        self.module.fail_json(msg='Could not fetch mapper %s for identity provider %s in realm %s: %s' % (mid, alias, realm, str(e)))