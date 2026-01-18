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
def del_cached_iface(self, vsctl_iface):
    vsctl_iface.port().ifaces.remove(vsctl_iface)
    vsctl_iface.port = None
    del self.ifaces[vsctl_iface.iface_cfg.name]
    vsctl_iface.iface_cfg.delete()