from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def create_clientscope_protocolmapper(self, cid, mapper_rep, realm='master'):
    """ Create a Keycloak clientscope protocolmapper.

        :param cid: Id of the clientscope.
        :param mapper_rep: a ProtocolMapperRepresentation of the protocolmapper to be created. Must contain at minimum the field name.
        :return: HTTPResponse object on success
        """
    protocolmappers_url = URL_CLIENTSCOPE_PROTOCOLMAPPERS.format(url=self.baseurl, id=cid, realm=realm)
    try:
        return open_url(protocolmappers_url, method='POST', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(mapper_rep), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not create protocolmapper %s in realm %s: %s' % (mapper_rep['name'], realm, str(e)))