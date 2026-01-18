import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def clear_db_attribute(self, table, record, column):
    """
        Clears values from 'column' in 'record' in 'table'.

        This method is corresponding to the following ovs-vsctl command::

            $ ovs-vsctl clear TBL REC COL
        """
    command = ovs_vsctl.VSCtlCommand('clear', (table, record, column))
    self.run_command([command])