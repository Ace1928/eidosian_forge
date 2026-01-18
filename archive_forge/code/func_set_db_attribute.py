import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def set_db_attribute(self, table, record, column, value, key=None):
    """
        Sets 'value' into 'column' in 'record' in 'table'.

        This method is corresponding to the following ovs-vsctl command::

            $ ovs-vsctl set TBL REC COL[:KEY]=VALUE
        """
    if key is not None:
        column = '%s:%s' % (column, key)
    command = ovs_vsctl.VSCtlCommand('set', (table, record, '%s=%s' % (column, value)))
    self.run_command([command])