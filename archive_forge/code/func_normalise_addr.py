import sys as _sys
from xml.sax import make_parser, handler
from netaddr.core import Publisher, Subscriber
from netaddr.ip import IPAddress, IPNetwork, IPRange, cidr_abbrev_to_verbose
from netaddr.compat import _open_binary
def normalise_addr(self, addr):
    """
        Removes variations from address entries found in this particular file.
        """
    if '-' in addr:
        a1, a2 = addr.split('-')
        o1 = a1.strip().split('.')
        o2 = a2.strip().split('.')
        return '%s-%s' % ('.'.join([str(int(i)) for i in o1]), '.'.join([str(int(i)) for i in o2]))
    else:
        o1 = addr.strip().split('.')
        return '.'.join([str(int(i)) for i in o1])