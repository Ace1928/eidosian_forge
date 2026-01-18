from paste.util import intset
import socket
def iteraddresses(self):
    """Returns an iterator which iterates over ips in this iprange. An
        IP is returned in string form (e.g. '1.2.3.4')."""
    for v in super(IP4Range, self).__iter__():
        yield self._int2ip(v)