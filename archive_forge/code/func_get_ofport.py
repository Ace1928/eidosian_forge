import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def get_ofport(self, port_name):
    """
        Gets the OpenFlow port number.

        This method is corresponding to the following ovs-vsctl command::

            $ ovs-vsctl get Interface <port> ofport
        """
    ofport_list = self.db_get_val('Interface', port_name, 'ofport')
    assert len(ofport_list) == 1
    return int(ofport_list[0])