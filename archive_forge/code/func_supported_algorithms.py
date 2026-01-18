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
def supported_algorithms(self, entity_id):
    """
        Get all supported algorithms for an entry in the metadata.

        Example return data:

        {'digest_methods': ['http://www.w3.org/2001/04/xmldsig-more#sha224', 'http://www.w3.org/2001/04/xmlenc#sha256'],
         'signing_methods': ['http://www.w3.org/2001/04/xmldsig-more#rsa-sha256']}

        :param entity_id: Entity id
        :return: dict with keys and value-lists from metadata

        :type entity_id: string
        :rtype: dict
        """
    res = {'digest_methods': [], 'signing_methods': []}
    try:
        ext = self.__getitem__(entity_id)['extensions']
    except KeyError:
        return res
    for elem in ext['extension_elements']:
        if elem['__class__'] == classnames['algsupport_digest_method']:
            res['digest_methods'].append(elem['algorithm'])
        elif elem['__class__'] == classnames['algsupport_signing_method']:
            res['signing_methods'].append(elem['algorithm'])
    return res