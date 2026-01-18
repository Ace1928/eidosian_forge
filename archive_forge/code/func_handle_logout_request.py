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
def handle_logout_request(self, request, name_id, binding, sign=None, sign_alg=None, digest_alg=None, relay_state=None, sigalg=None, signature=None):
    """
        Deal with a LogoutRequest

        :param request: The request as text string
        :param name_id: The id of the current user
        :param binding: Which binding the message came in over
        :param sign: Whether the response will be signed or not
        :param sign_alg: The signing algorithm for the response
        :param digest_alg: The digest algorithm for the the response
        :param relay_state: The relay state of the request
        :param sigalg: The SigAlg query param of the request
        :param signature: The Signature query param of the request
        :return: Keyword arguments which can be used to send the response
            what's returned follow different patterns for different bindings.
            If the binding is BINDIND_SOAP, what is returned looks like this::

                {
                    "data": <the SOAP enveloped response>
                    "url": "",
                    'headers': [('content-type', 'application/soap+xml')]
                    'method': "POST
                }
        """
    logger.debug('logout request: %s', request)
    _req = self.parse_logout_request(xmlstr=request, binding=binding, relay_state=relay_state, sigalg=sigalg, signature=signature)
    if _req.message.name_id == name_id:
        try:
            if self.local_logout(name_id):
                status = success_status_factory()
            else:
                status = status_message_factory('Server error', STATUS_REQUEST_DENIED)
        except KeyError:
            status = status_message_factory('Server error', STATUS_REQUEST_DENIED)
    else:
        status = status_message_factory('Wrong user', STATUS_UNKNOWN_PRINCIPAL)
    response_bindings = {BINDING_SOAP: [BINDING_SOAP], BINDING_HTTP_POST: [BINDING_HTTP_POST, BINDING_HTTP_REDIRECT], BINDING_HTTP_REDIRECT: [BINDING_HTTP_REDIRECT, BINDING_HTTP_POST]}.get(binding, [])
    for response_binding in response_bindings:
        sign = sign if sign is not None else self.logout_responses_signed
        sign_redirect = sign and response_binding == BINDING_HTTP_REDIRECT
        sign_post = sign and (not sign_redirect)
        try:
            response = self.create_logout_response(_req.message, bindings=[response_binding], status=status, sign=sign_post, sign_alg=sign_alg, digest_alg=digest_alg)
            rinfo = self.response_args(_req.message, [response_binding])
            return self.apply_binding(rinfo['binding'], response, rinfo['destination'], relay_state, response=True, sign=sign_redirect, sigalg=sign_alg)
        except Exception:
            continue
    log_ctx = {'message': 'No supported bindings found to create LogoutResponse', 'issuer': _req.issuer.text, 'response_bindings': response_bindings}
    raise SAMLError(log_ctx)