from saml2 import extension_elements_to_elements
from saml2.authn_context import ippword
from saml2.authn_context import mobiletwofactor
from saml2.authn_context import ppt
from saml2.authn_context import pword
from saml2.authn_context import sslcert
from saml2.saml import AuthnContext
from saml2.saml import AuthnContextClassRef
from saml2.samlp import RequestedAuthnContext
def pick(self, req_authn_context=None):
    """
        Given the authentication context find zero or more places where
        the user could be sent next. Ordered according to security level.

        :param req_authn_context: The requested context as an
            RequestedAuthnContext instance
        :return: An URL
        """
    if req_authn_context is None:
        return self._pick_by_class_ref(UNSPECIFIED, 'minimum')
    if req_authn_context.authn_context_class_ref:
        if req_authn_context.comparison:
            _cmp = req_authn_context.comparison
        else:
            _cmp = 'exact'
        if _cmp == 'exact':
            res = []
            for cls_ref in req_authn_context.authn_context_class_ref:
                res += self._pick_by_class_ref(cls_ref.text, _cmp)
            return res
        else:
            return self._pick_by_class_ref(req_authn_context.authn_context_class_ref[0].text, _cmp)
    elif req_authn_context.authn_context_decl_ref:
        if req_authn_context.comparison:
            _cmp = req_authn_context.comparison
        else:
            _cmp = 'exact'
        return self._pick_by_class_ref(req_authn_context.authn_context_decl_ref, _cmp)