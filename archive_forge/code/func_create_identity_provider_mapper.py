from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def create_identity_provider_mapper(self, mapper, alias, realm='master'):
    """ Create an identity provider mapper.
        :param mapper: IdentityProviderMapperRepresentation of the mapper to be created.
        :param alias: Alias of the identity provider.
        :param realm: Realm in which this identity provider resides, default "master".
        :return: HTTPResponse object on success
        """
    mappers_url = URL_IDENTITY_PROVIDER_MAPPERS.format(url=self.baseurl, realm=realm, alias=alias)
    try:
        return open_url(mappers_url, method='POST', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(mapper), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not create identity provider mapper %s for idp %s in realm %s: %s' % (mapper['name'], alias, realm, str(e)))