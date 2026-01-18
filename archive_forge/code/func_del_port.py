import logging
import operator
import os
import re
import sys
import weakref
import ovs.db.data
import ovs.db.parser
import ovs.db.schema
import ovs.db.types
import ovs.poller
import ovs.json
from ovs import jsonrpc
from ovs import ovsuuid
from ovs import stream
from ovs.db import idl
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.ovs import vswitch_idl
from os_ken.lib.stringify import StringifyMixin
def del_port(self, vsctl_port):
    if vsctl_port.bridge().parent:
        ovsrec_bridge = vsctl_port.bridge().parent.br_cfg
    else:
        ovsrec_bridge = vsctl_port.bridge().br_cfg
    self.bridge_delete_port(ovsrec_bridge, vsctl_port.port_cfg)
    for vsctl_iface in vsctl_port.ifaces.copy():
        self.del_cached_iface(vsctl_iface)
    self.del_cached_port(vsctl_port)