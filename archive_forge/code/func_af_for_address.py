import socket
import dns.ipv4
import dns.ipv6
from ._compat import maybe_ord
def af_for_address(text):
    """Determine the address family of a textual-form network address.

    *text*, a ``text``, the textual address.

    Raises ``ValueError`` if the address family cannot be determined
    from the input.

    Returns an ``int``.
    """
    try:
        dns.ipv4.inet_aton(text)
        return AF_INET
    except Exception:
        try:
            dns.ipv6.inet_aton(text)
            return AF_INET6
        except:
            raise ValueError