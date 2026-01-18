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
def create_authn_response(self, identity, in_response_to, destination, sp_entity_id, name_id_policy=None, userid=None, name_id=None, authn=None, issuer=None, sign_response=None, sign_assertion=None, encrypt_cert_advice=None, encrypt_cert_assertion=None, encrypt_assertion=None, encrypt_assertion_self_contained=True, encrypted_advice_attributes=False, pefim=False, sign_alg=None, digest_alg=None, session_not_on_or_after=None, **kwargs):
    """Constructs an AuthenticationResponse

        :param identity: Information about an user
        :param in_response_to: The identifier of the authentication request
            this response is an answer to.
        :param destination: Where the response should be sent
        :param sp_entity_id: The entity identifier of the Service Provider
        :param name_id_policy: How the NameID should be constructed
        :param userid: The subject identifier
        :param name_id: The identifier of the subject. A saml.NameID instance.
        :param authn: Dictionary with information about the authentication
            context
        :param issuer: Issuer of the response
        :param sign_assertion: Whether the assertion should be signed or not.
        :param sign_response: Whether the response should be signed or not.
        :param encrypt_assertion: True if assertions should be encrypted.
        :param encrypt_assertion_self_contained: True if all encrypted
        assertions should have alla namespaces
        selfcontained.
        :param encrypted_advice_attributes: True if assertions in the advice
        element should be encrypted.
        :param encrypt_cert_advice: Certificate to be used for encryption of
        assertions in the advice element.
        :param encrypt_cert_assertion: Certificate to be used for encryption
        of assertions.
        :param pefim: True if a response according to the PEFIM profile
        should be created.
        :return: A response instance
        """
    try:
        args = self.gather_authn_response_args(sp_entity_id, name_id_policy=name_id_policy, userid=userid, name_id=name_id, sign_response=sign_response, sign_assertion=sign_assertion, encrypt_cert_advice=encrypt_cert_advice, encrypt_cert_assertion=encrypt_cert_assertion, encrypt_assertion=encrypt_assertion, encrypt_assertion_self_contained=encrypt_assertion_self_contained, encrypted_advice_attributes=encrypted_advice_attributes, pefim=pefim, **kwargs)
    except OSError as exc:
        response = self.create_error_response(in_response_to, destination=destination, info=exc, sign=sign_response, sign_alg=sign_alg, digest_alg=digest_alg)
        return str(response).split('\n')
    try:
        _authn = authn
        return self._authn_response(in_response_to, destination, sp_entity_id, identity, authn=_authn, issuer=issuer, pefim=pefim, sign_alg=sign_alg, digest_alg=digest_alg, session_not_on_or_after=session_not_on_or_after, **args)
    except MissingValue as exc:
        return self.create_error_response(in_response_to, destination=destination, info=exc, sign=sign_response, sign_alg=sign_alg, digest_alg=digest_alg)