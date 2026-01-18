import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def get_port_name_list(self):
    """
        Gets a list of all ports on OVS instance.

        This method is corresponding to the following ovs-vsctl command::

            $ ovs-vsctl list-ports <bridge>
        """
    command = ovs_vsctl.VSCtlCommand('list-ports', (self.br_name,))
    self.run_command([command])
    return command.result