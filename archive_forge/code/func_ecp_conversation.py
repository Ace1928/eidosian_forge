from http import cookiejar as cookielib
import logging
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import saml
from saml2 import samlp
from saml2 import soap
from saml2.client_base import MIME_PAOS
from saml2.config import Config
from saml2.entity import Entity
from saml2.httpbase import dict2set_list
from saml2.httpbase import set_list2dict
from saml2.mdstore import MetadataStore
from saml2.profile import ecp
from saml2.profile import paos
from saml2.s_utils import BadRequest
def ecp_conversation(self, respdict, idp_entity_id=None):
    """

        :param respdict:
        :param idp_entity_id:
        :return:
        """
    args = self.parse_sp_ecp_response(respdict)
    idp_response = self.phase2(idp_entity_id=idp_entity_id, **args)
    ht_args = self.use_soap(idp_response, args['rc_url'], [args['relay_state']])
    ht_args['headers'][0] = ('Content-Type', MIME_PAOS)
    logger.debug('[P3] Post to SP: %s', ht_args['data'])
    response = self.send(**ht_args)
    if response.status_code == 302:
        pass
    else:
        raise SAMLError(f'Error POSTing package to SP: {response.text}')
    logger.debug('[P3] SP response: %s', response.text)
    self.done_ecp = True
    logger.debug('Done ECP')
    return None