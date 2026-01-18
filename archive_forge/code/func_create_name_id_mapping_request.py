import logging
import threading
import time
from typing import Mapping
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlparse
from warnings import warn as _warn
import saml2
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_PAOS
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import saml
from saml2 import samlp
from saml2 import soap
from saml2.entity import Entity
from saml2.extension import sp_type
from saml2.extension.requested_attributes import RequestedAttribute
from saml2.extension.requested_attributes import RequestedAttributes
from saml2.mdstore import locations
from saml2.population import Population
from saml2.profile import ecp
from saml2.profile import paos
from saml2.response import AssertionIDResponse
from saml2.response import AttributeResponse
from saml2.response import AuthnQueryResponse
from saml2.response import AuthnResponse
from saml2.response import AuthzResponse
from saml2.response import NameIDMappingResponse
from saml2.response import StatusError
from saml2.s_utils import UnravelError
from saml2.s_utils import do_attributes
from saml2.s_utils import signature
from saml2.saml import NAMEID_FORMAT_PERSISTENT
from saml2.saml import NAMEID_FORMAT_TRANSIENT
from saml2.saml import AuthnContextClassRef
from saml2.samlp import AttributeQuery
from saml2.samlp import AuthnQuery
from saml2.samlp import AuthnRequest
from saml2.samlp import AuthzDecisionQuery
from saml2.samlp import Extensions
from saml2.samlp import NameIDMappingRequest
from saml2.samlp import RequestedAuthnContext
from saml2.soap import make_soap_enveloped_saml_thingy
def create_name_id_mapping_request(self, name_id_policy, name_id=None, base_id=None, encrypted_id=None, destination=None, message_id=0, consent=None, extensions=None, sign=None, nsprefix=None, sign_alg=None, digest_alg=None):
    """

        :param name_id_policy:
        :param name_id:
        :param base_id:
        :param encrypted_id:
        :param destination:
        :param message_id: Message identifier
        :param consent: If the principal gave her consent to this request
        :param extensions: Possible request extensions
        :param sign: Whether the request should be signed or not.
        :return:
        """
    if not name_id and (not base_id) and (not encrypted_id):
        raise ValueError('At least one of name_id, base_id or encrypted_id must be present.')
    id_attr = {'name_id': name_id, 'base_id': base_id if not name_id else None, 'encrypted_id': encrypted_id if not name_id and (not base_id) else None}
    return self._message(NameIDMappingRequest, destination, message_id, consent, extensions, sign, name_id_policy=name_id_policy, **id_attr, nsprefix=nsprefix, sign_alg=sign_alg, digest_alg=digest_alg)