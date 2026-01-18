import os
import re
import uuid
from urllib.parse import urlparse
from tornado.escape import url_escape
from ..base.handlers import JupyterHandler
from .decorator import allow_unauthenticated
from .security import passwd_check, set_password
@classmethod
def get_login_available(cls, settings):
    """DEPRECATED in 2.0, use IdentityProvider API"""
    return bool(cls.password_from_settings(settings) or settings.get('token'))