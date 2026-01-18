from io import BytesIO
from .lazy_import import lazy_import
from breezy.i18n import gettext
from . import errors, urlutils
from .trace import note
from .transport import (do_catching_redirections, get_transport,
def redirected_transport(transport, exception, redirection_notice):
    note(redirection_notice)
    url, filename = urlutils.split(exception.target, exclude_trailing_slash=False)
    if not filename:
        raise errors.NotABundle(gettext('A directory cannot be a bundle'))
    return get_transport_from_url(url)