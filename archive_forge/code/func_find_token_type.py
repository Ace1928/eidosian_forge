from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.common import Request
from .base import BaseEndpoint, catch_errors_and_unavailability
def find_token_type(self, request):
    """Token type identification.

        RFC 6749 does not provide a method for easily differentiating between
        different token types during protected resource access. We estimate
        the most likely token type (if any) by asking each known token type
        to give an estimation based on the request.
        """
    estimates = sorted(((t.estimate_type(request), n) for n, t in self.tokens.items()), reverse=True)
    return estimates[0][1] if len(estimates) else None