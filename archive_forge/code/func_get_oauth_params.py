from __future__ import absolute_import, unicode_literals
import base64
import hashlib
import logging
import sys
from oauthlib.common import Request, urlencode, generate_nonce
from oauthlib.common import generate_timestamp, to_unicode
from . import parameters, signature
def get_oauth_params(self, request):
    """Get the basic OAuth parameters to be used in generating a signature."""
    nonce = generate_nonce() if self.nonce is None else self.nonce
    timestamp = generate_timestamp() if self.timestamp is None else self.timestamp
    params = [('oauth_nonce', nonce), ('oauth_timestamp', timestamp), ('oauth_version', '1.0'), ('oauth_signature_method', self.signature_method), ('oauth_consumer_key', self.client_key)]
    if self.resource_owner_key:
        params.append(('oauth_token', self.resource_owner_key))
    if self.callback_uri:
        params.append(('oauth_callback', self.callback_uri))
    if self.verifier:
        params.append(('oauth_verifier', self.verifier))
    content_type = request.headers.get('Content-Type', None)
    content_type_eligible = content_type and content_type.find('application/x-www-form-urlencoded') < 0
    if request.body is not None and content_type_eligible:
        params.append(('oauth_body_hash', base64.b64encode(hashlib.sha1(request.body.encode('utf-8')).digest()).decode('utf-8')))
    return params