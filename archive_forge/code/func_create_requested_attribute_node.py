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
def create_requested_attribute_node(requested_attrs, attribute_converters):
    items = []
    for attr in requested_attrs:
        friendly_name = attr.get('friendly_name')
        name = attr.get('name')
        name_format = attr.get('name_format')
        is_required = str(attr.get('required', False)).lower()
        if not name and (not friendly_name):
            raise ValueError("Missing required attribute: 'name' or 'friendly_name'")
        if not name:
            for converter in attribute_converters:
                try:
                    name = converter._to[friendly_name.lower()]
                except KeyError:
                    continue
                else:
                    if not name_format:
                        name_format = converter.name_format
                    break
        if not friendly_name:
            for converter in attribute_converters:
                try:
                    friendly_name = converter._fro[name.lower()]
                except KeyError:
                    continue
                else:
                    if not name_format:
                        name_format = converter.name_format
                    break
        items.append(RequestedAttribute(is_required=is_required, name_format=name_format, friendly_name=friendly_name, name=name))
    node = RequestedAttributes(extension_elements=items)
    return node