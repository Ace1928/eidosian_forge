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
@staticmethod
def is_session_valid(_session_id):
    """Place holder. Supposed to check if the session is still valid."""
    return True