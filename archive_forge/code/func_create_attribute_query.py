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
def create_attribute_query(self, destination, name_id=None, attribute=None, message_id=0, consent=None, extensions=None, sign=None, sign_prepare=None, sign_alg=None, digest_alg=None, **kwargs):
    """Constructs an AttributeQuery

        :param destination: To whom the query should be sent
        :param name_id: The identifier of the subject
        :param attribute: A dictionary of attributes and values that is
            asked for. The key are one of 4 variants:
            3-tuple of name_format,name and friendly_name,
            2-tuple of name_format and name,
            1-tuple with name or
            just the name as a string.
        :param sp_name_qualifier: The unique identifier of the
            service provider or affiliation of providers for whom the
            identifier was generated.
        :param name_qualifier: The unique identifier of the identity
            provider that generated the identifier.
        :param format: The format of the name ID
        :param message_id: The identifier of the session
        :param consent: Whether the principal have given her consent
        :param extensions: Possible extensions
        :param sign: Whether the query should be signed or not.
        :param sign_prepare: Whether the Signature element should be added.
        :return: Tuple of request ID and an AttributeQuery instance
        """
    if name_id is None:
        if 'subject_id' in kwargs:
            name_id = saml.NameID(text=kwargs['subject_id'])
            for key in ['sp_name_qualifier', 'name_qualifier', 'format']:
                try:
                    setattr(name_id, key, kwargs[key])
                except KeyError:
                    pass
        else:
            raise AttributeError('Missing required parameter')
    elif isinstance(name_id, str):
        name_id = saml.NameID(text=name_id)
        for key in ['sp_name_qualifier', 'name_qualifier', 'format']:
            try:
                setattr(name_id, key, kwargs[key])
            except KeyError:
                pass
    subject = saml.Subject(name_id=name_id)
    if attribute:
        attribute = do_attributes(attribute)
    try:
        nsprefix = kwargs['nsprefix']
    except KeyError:
        nsprefix = None
    return self._message(AttributeQuery, destination, message_id, consent, extensions, sign, sign_prepare, subject=subject, attribute=attribute, nsprefix=nsprefix, sign_alg=sign_alg, digest_alg=digest_alg)