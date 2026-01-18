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
def prepare_for_authenticate(self, entityid=None, relay_state='', binding=saml2.BINDING_HTTP_REDIRECT, vorg='', nameid_format=None, scoping=None, consent=None, extensions=None, sign=None, sigalg=None, digest_alg=None, response_binding=saml2.BINDING_HTTP_POST, **kwargs):
    """Makes all necessary preparations for an authentication request.

        :param entityid: The entity ID of the IdP to send the request to
        :param relay_state: To where the user should be returned after
            successfull log in.
        :param binding: Which binding to use for sending the request
        :param vorg: The entity_id of the virtual organization I'm a member of
        :param nameid_format:
        :param scoping: For which IdPs this query are aimed.
        :param consent: Whether the principal have given her consent
        :param extensions: Possible extensions
        :param sign: Whether the request should be signed or not.
        :param response_binding: Which binding to use for receiving the response
        :param kwargs: Extra key word arguments
        :return: session id and AuthnRequest info
        """
    reqid, negotiated_binding, info = self.prepare_for_negotiated_authenticate(entityid=entityid, relay_state=relay_state, binding=binding, vorg=vorg, nameid_format=nameid_format, scoping=scoping, consent=consent, extensions=extensions, sign=sign, sigalg=sigalg, digest_alg=digest_alg, response_binding=response_binding, **kwargs)
    if negotiated_binding != binding:
        raise ValueError(f"Negotiated binding '{negotiated_binding}' does not match binding to use '{binding}'")
    return (reqid, info)