from __future__ import absolute_import, unicode_literals
import base64
import hashlib
import logging
import sys
from oauthlib.common import Request, urlencode, generate_nonce
from oauthlib.common import generate_timestamp, to_unicode
from . import parameters, signature
def get_oauth_signature(self, request):
    """Get an OAuth signature to be used in signing a request

        To satisfy `section 3.4.1.2`_ item 2, if the request argument's
        headers dict attribute contains a Host item, its value will
        replace any netloc part of the request argument's uri attribute
        value.

        .. _`section 3.4.1.2`:
        https://tools.ietf.org/html/rfc5849#section-3.4.1.2
        """
    if self.signature_method == SIGNATURE_PLAINTEXT:
        return signature.sign_plaintext(self.client_secret, self.resource_owner_secret)
    uri, headers, body = self._render(request)
    collected_params = signature.collect_parameters(uri_query=urlparse.urlparse(uri).query, body=body, headers=headers)
    log.debug('Collected params: {0}'.format(collected_params))
    normalized_params = signature.normalize_parameters(collected_params)
    normalized_uri = signature.normalize_base_string_uri(uri, headers.get('Host', None))
    log.debug('Normalized params: {0}'.format(normalized_params))
    log.debug('Normalized URI: {0}'.format(normalized_uri))
    base_string = signature.construct_base_string(request.http_method, normalized_uri, normalized_params)
    log.debug('Signing: signature base string: {0}'.format(base_string))
    if self.signature_method not in self.SIGNATURE_METHODS:
        raise ValueError('Invalid signature method.')
    sig = self.SIGNATURE_METHODS[self.signature_method](base_string, self)
    log.debug('Signature: {0}'.format(sig))
    return sig