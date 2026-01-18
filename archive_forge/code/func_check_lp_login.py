from ... import errors, trace, transport
from ...config import AuthenticationConfig, GlobalStack
from ...i18n import gettext
def check_lp_login(username, _transport=None):
    """Check whether the given Launchpad username is okay.

    This will check for both existence and whether the user has
    uploaded SSH keys.
    """
    if _transport is None:
        _transport = transport.get_transport_from_url(LAUNCHPAD_BASE)
    try:
        data = _transport.get_bytes('~%s/+sshkeys' % username)
    except transport.NoSuchFile:
        raise UnknownLaunchpadUsername(user=username)
    if not data:
        raise NoRegisteredSSHKeys(user=username)