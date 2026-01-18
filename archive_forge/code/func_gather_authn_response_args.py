import dbm
import importlib
import logging
import shelve
import threading
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import class_name
from saml2 import element_to_extension_element
from saml2 import saml
from saml2.argtree import add_path
from saml2.argtree import is_set
from saml2.assertion import Assertion
from saml2.assertion import Policy
from saml2.assertion import filter_attribute_value_assertions
from saml2.assertion import restriction_from_attribute_spec
import saml2.cryptography.symmetric
from saml2.entity import Entity
from saml2.eptid import Eptid
from saml2.eptid import EptidShelve
from saml2.ident import IdentDB
from saml2.ident import decode
from saml2.profile import ecp
from saml2.request import AssertionIDRequest
from saml2.request import AttributeQuery
from saml2.request import AuthnQuery
from saml2.request import AuthnRequest
from saml2.request import AuthzDecisionQuery
from saml2.request import NameIDMappingRequest
from saml2.s_utils import MissingValue
from saml2.s_utils import Unknown
from saml2.s_utils import rndstr
from saml2.samlp import NameIDMappingResponse
from saml2.schema import soapenv
from saml2.sdb import SessionStorage
from saml2.sigver import CertificateError
from saml2.sigver import pre_signature_part
from saml2.sigver import signed_instance_factory
def gather_authn_response_args(self, sp_entity_id, name_id_policy, userid, **kwargs):
    kwargs['policy'] = kwargs.get('release_policy')
    args = {}
    param_defaults = {'policy': None, 'best_effort': False, 'sign_assertion': False, 'sign_response': False, 'encrypt_assertion': False, 'encrypt_assertion_self_contained': True, 'encrypted_advice_attributes': False, 'encrypt_cert_advice': None, 'encrypt_cert_assertion': None}
    for param, val_default in param_defaults.items():
        val_kw = kwargs.get(param)
        val_config = self.config.getattr(param, 'idp')
        args[param] = val_kw if val_kw is not None else val_config if val_config is not None else val_default
    for arg, attr, eca, pefim in [('encrypted_advice_attributes', 'verify_encrypt_cert_advice', 'encrypt_cert_advice', kwargs['pefim']), ('encrypt_assertion', 'verify_encrypt_cert_assertion', 'encrypt_cert_assertion', False)]:
        if args[arg] or pefim:
            _enc_cert = self.config.getattr(attr, 'idp')
            if _enc_cert is not None:
                if kwargs[eca] is None:
                    raise CertificateError('No SPCertEncType certificate for encryption contained in authentication request.')
                if not _enc_cert(kwargs[eca]):
                    raise CertificateError('Invalid certificate for encryption!')
    if 'name_id' not in kwargs or not kwargs['name_id']:
        nid_formats = []
        for _sp in self.metadata[sp_entity_id]['spsso_descriptor']:
            if 'name_id_format' in _sp:
                nid_formats.extend([n['text'] for n in _sp['name_id_format']])
        try:
            snq = name_id_policy.sp_name_qualifier
        except AttributeError:
            snq = sp_entity_id
        if not snq:
            snq = sp_entity_id
        kwa = {'sp_name_qualifier': snq}
        try:
            kwa['format'] = name_id_policy.format
        except AttributeError:
            pass
        _nids = self.ident.find_nameid(userid, **kwa)
        if _nids:
            args['name_id'] = _nids[0]
        else:
            args['name_id'] = self.ident.construct_nameid(userid, args['policy'], sp_entity_id, name_id_policy)
            logger.debug('construct_nameid: %s => %s', userid, args['name_id'])
    else:
        args['name_id'] = kwargs['name_id']
    for param in ['status', 'farg']:
        try:
            args[param] = kwargs[param]
        except KeyError:
            pass
    return args