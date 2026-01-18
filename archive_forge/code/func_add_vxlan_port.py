import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def add_vxlan_port(self, name, remote_ip, local_ip=None, key=None, ofport=None):
    """
        Creates a VxLAN tunnel port.

        See the description of ``add_tunnel_port()``.
        """
    self.add_tunnel_port(name, 'vxlan', remote_ip, local_ip=local_ip, key=key, ofport=ofport)