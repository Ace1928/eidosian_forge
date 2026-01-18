import os
import re
import uuid
from urllib.parse import urlparse
from tornado.escape import url_escape
from ..base.handlers import JupyterHandler
from .decorator import allow_unauthenticated
from .security import passwd_check, set_password
@classmethod
def get_user_token(cls, handler):
    """DEPRECATED in 2.0, use IdentityProvider API"""
    token = handler.token
    if not token:
        return None
    user_token = cls.get_token(handler)
    authenticated = False
    if user_token == token:
        handler.log.debug('Accepting token-authenticated connection from %s', handler.request.remote_ip)
        authenticated = True
    if authenticated:
        user_id = cls.get_user_cookie(handler)
        if user_id is None:
            user_id = uuid.uuid4().hex
            handler.log.info(f'Generating new user_id for token-authenticated request: {user_id}')
        return user_id
    else:
        return None