import sys as _sys
from xml.sax import make_parser, handler
from netaddr.core import Publisher, Subscriber
from netaddr.ip import IPAddress, IPNetwork, IPRange, cidr_abbrev_to_verbose
from netaddr.compat import _open_binary
def load_info():
    """
    Parse and load internal IANA data lookups with the latest information from
    data files.
    """
    ipv4 = IPv4Parser(_open_binary(__package__, 'ipv4-address-space.xml'))
    ipv4.attach(DictUpdater(IANA_INFO['IPv4'], 'IPv4', 'prefix'))
    ipv4.parse()
    ipv6 = IPv6Parser(_open_binary(__package__, 'ipv6-address-space.xml'))
    ipv6.attach(DictUpdater(IANA_INFO['IPv6'], 'IPv6', 'prefix'))
    ipv6.parse()
    ipv6ua = IPv6UnicastParser(_open_binary(__package__, 'ipv6-unicast-address-assignments.xml'))
    ipv6ua.attach(DictUpdater(IANA_INFO['IPv6_unicast'], 'IPv6_unicast', 'prefix'))
    ipv6ua.parse()
    mcast = MulticastParser(_open_binary(__package__, 'multicast-addresses.xml'))
    mcast.attach(DictUpdater(IANA_INFO['multicast'], 'multicast', 'address'))
    mcast.parse()