from .exceptions import OIDCNoPrompt
import datetime
import logging
from json import loads
from oauthlib.oauth2.rfc6749.errors import ConsentRequired, InvalidRequestError, LoginRequired
def add_id_token(self, token, token_handler, request):
    if not request.scopes or 'openid' not in request.scopes:
        return token
    if request.response_type and 'id_token' not in request.response_type:
        return token
    if request.max_age:
        d = datetime.datetime.utcnow()
        token['auth_time'] = d.isoformat('T') + 'Z'
    token['id_token'] = self.request_validator.get_id_token(token, token_handler, request)
    return token