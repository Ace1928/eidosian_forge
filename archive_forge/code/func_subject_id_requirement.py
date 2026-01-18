import hashlib
from hashlib import sha1
import importlib
from itertools import chain
import json
import logging
import os
from os.path import isfile
from os.path import join
from re import compile as regex_compile
import sys
from warnings import warn as _warn
import requests
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import md
from saml2 import saml
from saml2 import samlp
from saml2 import xmldsig
from saml2 import xmlenc
from saml2.extension.algsupport import NAMESPACE as NS_ALGSUPPORT
from saml2.extension.algsupport import DigestMethod
from saml2.extension.algsupport import SigningMethod
from saml2.extension.idpdisc import BINDING_DISCO
from saml2.extension.idpdisc import DiscoveryResponse
from saml2.extension.mdattr import NAMESPACE as NS_MDATTR
from saml2.extension.mdattr import EntityAttributes
from saml2.extension.mdrpi import NAMESPACE as NS_MDRPI
from saml2.extension.mdrpi import RegistrationInfo
from saml2.extension.mdrpi import RegistrationPolicy
from saml2.extension.mdui import NAMESPACE as NS_MDUI
from saml2.extension.mdui import Description
from saml2.extension.mdui import DisplayName
from saml2.extension.mdui import InformationURL
from saml2.extension.mdui import Logo
from saml2.extension.mdui import PrivacyStatementURL
from saml2.extension.mdui import UIInfo
from saml2.extension.shibmd import NAMESPACE as NS_SHIBMD
from saml2.extension.shibmd import Scope
from saml2.httpbase import HTTPBase
from saml2.md import NAMESPACE as NS_MD
from saml2.md import ArtifactResolutionService
from saml2.md import EntitiesDescriptor
from saml2.md import EntityDescriptor
from saml2.md import NameIDMappingService
from saml2.md import SingleSignOnService
from saml2.mdie import to_dict
from saml2.s_utils import UnknownSystemEntity
from saml2.s_utils import UnsupportedBinding
from saml2.sigver import SignatureError
from saml2.sigver import security_context
from saml2.sigver import split_len
from saml2.time_util import add_duration
from saml2.time_util import before
from saml2.time_util import instant
from saml2.time_util import str_to_time
from saml2.time_util import valid
from saml2.validate import NotValid
from saml2.validate import valid_instance
def subject_id_requirement(self, entity_id):
    try:
        entity_attributes = self.entity_attributes(entity_id)
    except KeyError:
        return []
    subject_id_reqs = entity_attributes.get('urn:oasis:names:tc:SAML:profiles:subject-id:req') or []
    subject_id_req = next(iter(subject_id_reqs), None)
    if subject_id_req == 'any':
        return [{'__class__': 'urn:oasis:names:tc:SAML:2.0:metadata&RequestedAttribute', 'name': 'urn:oasis:names:tc:SAML:attribute:pairwise-id', 'name_format': 'urn:oasis:names:tc:SAML:2.0:attrname-format:uri', 'friendly_name': 'pairwise-id', 'is_required': 'true'}, {'__class__': 'urn:oasis:names:tc:SAML:2.0:metadata&RequestedAttribute', 'name': 'urn:oasis:names:tc:SAML:attribute:subject-id', 'name_format': 'urn:oasis:names:tc:SAML:2.0:attrname-format:uri', 'friendly_name': 'subject-id', 'is_required': 'true'}]
    elif subject_id_req == 'pairwise-id':
        return [{'__class__': 'urn:oasis:names:tc:SAML:2.0:metadata&RequestedAttribute', 'name': 'urn:oasis:names:tc:SAML:attribute:pairwise-id', 'name_format': 'urn:oasis:names:tc:SAML:2.0:attrname-format:uri', 'friendly_name': 'pairwise-id', 'is_required': 'true'}]
    elif subject_id_req == 'subject-id':
        return [{'__class__': 'urn:oasis:names:tc:SAML:2.0:metadata&RequestedAttribute', 'name': 'urn:oasis:names:tc:SAML:attribute:subject-id', 'name_format': 'urn:oasis:names:tc:SAML:2.0:attrname-format:uri', 'friendly_name': 'subject-id', 'is_required': 'true'}]
    return []