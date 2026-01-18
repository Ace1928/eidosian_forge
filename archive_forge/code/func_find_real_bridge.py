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
def find_real_bridge(self, name, must_exist):
    vsctl_bridge = self.find_bridge(name, must_exist)
    if vsctl_bridge and vsctl_bridge.parent:
        vsctl_fatal('%s is a fake bridge' % name)
    return vsctl_bridge