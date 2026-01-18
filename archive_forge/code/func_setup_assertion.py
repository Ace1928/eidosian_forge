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
def setup_assertion(self, authn, sp_entity_id, in_response_to, consumer_url, name_id, policy, _issuer, authn_statement, identity, best_effort, sign_response, farg=None, session_not_on_or_after=None, sign_alg=None, digest_alg=None, **kwargs):
    """
        Construct and return the Assertion

        :param authn: Authentication information
        :param sp_entity_id:
        :param in_response_to: The ID of the request this is an answer to
        :param consumer_url: The recipient of the assertion
        :param name_id: The NameID of the subject
        :param policy: Assertion policies
        :param _issuer: Issuer of the statement
        :param authn_statement: An AuthnStatement instance
        :param identity: Identity information about the Subject
        :param best_effort: Even if not the SPs demands can be met send a
            response.
        :param sign_response: Sign the response, only applicable if
            ErrorResponse
        :param kwargs: Extra keyword arguments
        :return: An Assertion instance
        """
    ast = Assertion(identity)
    ast.acs = self.config.getattr('attribute_converters')
    if policy is None:
        policy = Policy(mds=self.metadata)
    try:
        ast.apply_policy(sp_entity_id, policy)
    except MissingValue as exc:
        if not best_effort:
            response = self.create_error_response(in_response_to, destination=consumer_url, info=exc, sign=sign_response, sign_alg=sign_alg, digest_alg=digest_alg)
            return str(response).split('\n')
    farg = self.update_farg(in_response_to, consumer_url, farg)
    if authn:
        authn_args = {AUTHN_DICT_MAP[k]: v for k, v in authn.items() if k in AUTHN_DICT_MAP}
        authn_args.update(kwargs)
        assertion = ast.construct(sp_entity_id, self.config.attribute_converters, policy, issuer=_issuer, farg=farg['assertion'], name_id=name_id, session_not_on_or_after=session_not_on_or_after, **authn_args)
    elif authn_statement:
        assertion = ast.construct(sp_entity_id, self.config.attribute_converters, policy, issuer=_issuer, authn_statem=authn_statement, farg=farg['assertion'], name_id=name_id, **kwargs)
    else:
        assertion = ast.construct(sp_entity_id, self.config.attribute_converters, policy, issuer=_issuer, farg=farg['assertion'], name_id=name_id, session_not_on_or_after=session_not_on_or_after, **kwargs)
    return assertion