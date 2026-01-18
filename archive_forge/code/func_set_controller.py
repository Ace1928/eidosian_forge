import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def set_controller(self, controllers):
    """
        Sets the OpenFlow controller address.

        This method is corresponding to the following ovs-vsctl command::

            $ ovs-vsctl set-controller <bridge> <target>...
        """
    command = ovs_vsctl.VSCtlCommand('set-controller', [self.br_name])
    command.args.extend(controllers)
    self.run_command([command])