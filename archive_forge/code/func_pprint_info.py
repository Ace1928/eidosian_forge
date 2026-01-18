import sys as _sys
from xml.sax import make_parser, handler
from netaddr.core import Publisher, Subscriber
from netaddr.ip import IPAddress, IPNetwork, IPRange, cidr_abbrev_to_verbose
from netaddr.compat import _open_binary
def pprint_info(fh=None):
    """
    Pretty prints IANA information to filehandle.
    """
    if fh is None:
        fh = _sys.stdout
    for category in sorted(IANA_INFO):
        fh.write('-' * len(category) + '\n')
        fh.write(category + '\n')
        fh.write('-' * len(category) + '\n')
        ipranges = IANA_INFO[category]
        for iprange in sorted(ipranges):
            details = ipranges[iprange]
            fh.write('%-45r' % iprange + details + '\n')