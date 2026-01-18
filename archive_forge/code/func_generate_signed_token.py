from __future__ import absolute_import, unicode_literals
import collections
import datetime
import logging
import re
import sys
import time
def generate_signed_token(private_pem, request):
    import jwt
    now = datetime.datetime.utcnow()
    claims = {'scope': request.scope, 'exp': now + datetime.timedelta(seconds=request.expires_in)}
    claims.update(request.claims)
    token = jwt.encode(claims, private_pem, 'RS256')
    token = to_unicode(token, 'UTF-8')
    return token