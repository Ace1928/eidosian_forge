from base64 import urlsafe_b64encode
import hashlib
import json
import logging
from string import ascii_letters, digits
import webbrowser
import wsgiref.simple_server
import wsgiref.util
import google.auth.transport.requests
import google.oauth2.credentials
import google_auth_oauthlib.helpers
@redirect_uri.setter
def redirect_uri(self, value):
    """The OAuth 2.0 redirect URI. Pass-through to
        ``self.oauth2session.redirect_uri``."""
    self.oauth2session.redirect_uri = value