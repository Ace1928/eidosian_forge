import datetime
import os
import subprocess  # nosec : see comments in the code below
import uuid
from oslo_log import log
from oslo_utils import fileutils
from oslo_utils import importutils
from oslo_utils import timeutils
import saml2
from saml2 import client_base
from saml2 import md
from saml2.profile import ecp
from saml2 import saml
from saml2 import samlp
from saml2.schema import soapenv
from saml2 import sigver
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def samlize_token(self, issuer, recipient, user, user_domain_name, roles, project, project_domain_name, groups, expires_in=None):
    """Convert Keystone attributes to a SAML assertion.

        :param issuer: URL of the issuing party
        :type issuer: string
        :param recipient: URL of the recipient
        :type recipient: string
        :param user: User name
        :type user: string
        :param user_domain_name: User Domain name
        :type user_domain_name: string
        :param roles: List of role names
        :type roles: list
        :param project: Project name
        :type project: string
        :param project_domain_name: Project Domain name
        :type project_domain_name: string
        :param groups: List of strings of user groups and domain name, where
                       strings are serialized dictionaries.
        :type groups: list
        :param expires_in: Sets how long the assertion is valid for, in seconds
        :type expires_in: int

        :returns: XML <Response> object

        """
    expiration_time = self._determine_expiration_time(expires_in)
    status = self._create_status()
    saml_issuer = self._create_issuer(issuer)
    subject = self._create_subject(user, expiration_time, recipient)
    attribute_statement = self._create_attribute_statement(user, user_domain_name, roles, project, project_domain_name, groups)
    authn_statement = self._create_authn_statement(issuer, expiration_time)
    signature = self._create_signature()
    assertion = self._create_assertion(saml_issuer, signature, subject, authn_statement, attribute_statement)
    assertion = _sign_assertion(assertion)
    response = self._create_response(saml_issuer, status, assertion, recipient)
    return response