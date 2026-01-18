from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneclient import access
from keystoneclient import base
from keystoneclient.i18n import _
def calc_id(token):
    if isinstance(token, access.AccessInfo):
        return token.auth_token
    return base.getid(token)