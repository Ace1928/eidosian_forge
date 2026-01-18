import logging
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import time_util
from saml2.attribute_converter import to_local
from saml2.response import IncorrectlySigned
from saml2.s_utils import OtherError
from saml2.s_utils import VersionMismatch
from saml2.sigver import verify_redirect_signature
from saml2.validate import NotValid
from saml2.validate import valid_instance
def evidence(self):
    """The evidence on which the decision is based"""