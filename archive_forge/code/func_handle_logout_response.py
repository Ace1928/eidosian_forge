import logging
import saml2
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import saml
from saml2.client_base import Base
from saml2.client_base import LogoutError
from saml2.client_base import NoServiceDefined
from saml2.client_base import SignOnError
from saml2.httpbase import HTTPError
from saml2.ident import code
from saml2.ident import decode
from saml2.mdstore import locations
from saml2.s_utils import sid
from saml2.s_utils import status_message_factory
from saml2.s_utils import success_status_factory
from saml2.saml import AssertionIDRef
from saml2.samlp import STATUS_REQUEST_DENIED
from saml2.samlp import STATUS_UNKNOWN_PRINCIPAL
from saml2.time_util import not_on_or_after
def handle_logout_response(self, response, sign_alg=None, digest_alg=None):
    """handles a Logout response

        :param response: A response.Response instance
        :return: 4-tuple of (session_id of the last sent logout request,
            response message, response headers and message)
        """
    logger.debug('state: %s', self.state)
    status = self.state[response.in_response_to]
    logger.debug('status: %s', status)
    issuer = response.issuer()
    logger.debug('issuer: %s', issuer)
    del self.state[response.in_response_to]
    if status['entity_ids'] == [issuer]:
        self.local_logout(decode(status['name_id']))
        return (0, '200 Ok', [('Content-type', 'text/html')], [])
    else:
        status['entity_ids'].remove(issuer)
        if 'sign_alg' in status:
            sign_alg = status['sign_alg']
        return self.do_logout(decode(status['name_id']), status['entity_ids'], status['reason'], status['not_on_or_after'], status['sign'], sign_alg=sign_alg, digest_alg=digest_alg)