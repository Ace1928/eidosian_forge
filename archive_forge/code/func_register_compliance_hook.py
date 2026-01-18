import logging
from oauthlib.common import generate_token, urldecode
from oauthlib.oauth2 import WebApplicationClient, InsecureTransportError
from oauthlib.oauth2 import LegacyApplicationClient
from oauthlib.oauth2 import TokenExpiredError, is_secure_transport
import requests
def register_compliance_hook(self, hook_type, hook):
    """Register a hook for request/response tweaking.

        Available hooks are:
            access_token_response invoked before token parsing.
            refresh_token_response invoked before refresh token parsing.
            protected_request invoked before making a request.
            access_token_request invoked before making a token fetch request.
            refresh_token_request invoked before making a refresh request.

        If you find a new hook is needed please send a GitHub PR request
        or open an issue.
        """
    if hook_type not in self.compliance_hook:
        raise ValueError('Hook type %s is not in %s.', hook_type, self.compliance_hook)
    self.compliance_hook[hook_type].add(hook)