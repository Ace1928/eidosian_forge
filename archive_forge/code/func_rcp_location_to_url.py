import re
from . import urlutils
from .hooks import Hooks
def rcp_location_to_url(location, scheme='ssh'):
    """Convert a rcp-style location to a URL.

    :param location: Location to convert, e.g. "foo:bar"
    :param scheme: URL scheme to return, defaults to "ssh"
    :return: A URL, e.g. "ssh://foo/bar"
    :raises ValueError: if this is not a RCP-style URL
    """
    host, user, path = parse_rcp_location(location)
    quoted_user = urlutils.quote(user) if user else None
    url = urlutils.URL(scheme=scheme, quoted_user=quoted_user, port=None, quoted_password=None, quoted_host=urlutils.quote(host), quoted_path=urlutils.quote(path))
    return str(url)