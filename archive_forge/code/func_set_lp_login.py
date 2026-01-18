from ... import errors, trace, transport
from ...config import AuthenticationConfig, GlobalStack
from ...i18n import gettext
def set_lp_login(username, _config=None):
    """Set the user's Launchpad username"""
    _set_global_option(username, _config)
    if username is not None:
        _set_auth_user(username)