from . import exceptions
from . import misc
from . import normalizers
def host_is_valid(host, require=False):
    """Determine if the host string is valid.

    :param str host:
        The host to validate.
    :param bool require:
        (optional) Specify if host must not be None.
    :returns:
        ``True`` if valid, ``False`` otherwise
    :rtype:
        bool
    """
    validated = is_valid(host, misc.HOST_MATCHER, require)
    if validated and host is not None and misc.IPv4_MATCHER.match(host):
        return valid_ipv4_host_address(host)
    elif validated and host is not None and misc.IPv6_MATCHER.match(host):
        return misc.IPv6_NO_RFC4007_MATCHER.match(host) is not None
    return validated