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
def create_authn_query_response(self, subject, session_index=None, requested_context=None, in_response_to=None, issuer=None, sign_response=None, status=None, sign_alg=None, digest_alg=None, **kwargs):
    """
        A successful <Response> will contain one or more assertions containing
        authentication statements.

        :return:
        """
    margs = self.message_args()
    asserts = [saml.Assertion(authn_statement=statement, subject=subject, **margs) for statement in self.session_db.get_authn_statements(subject.name_id, session_index, requested_context)]
    if asserts:
        args = {'assertion': asserts}
    else:
        args = {}
    return self._response(in_response_to, '', status, issuer, sign_response, to_sign=[], sign_alg=sign_alg, digest_alg=digest_alg, **args)