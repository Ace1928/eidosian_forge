import logging
from os_ken.lib.mac import haddr_to_str
def port_add(self, dpid, port, mac):
    """
        :returns: old port if learned. (this may be = port)
                  None otherwise
        """
    old_port = self.mac_to_port[dpid].get(mac, None)
    self.mac_to_port[dpid][mac] = port
    if old_port is not None and old_port != port:
        LOG.debug('port_add: 0x%016x 0x%04x %s', dpid, port, haddr_to_str(mac))
    return old_port