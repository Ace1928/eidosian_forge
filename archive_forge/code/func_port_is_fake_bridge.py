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
@staticmethod
def port_is_fake_bridge(ovsrec_port):
    tag = ovsrec_port.tag
    if isinstance(tag, list):
        if len(tag) == 0:
            tag = 0
        else:
            tag = tag[0]
    return ovsrec_port.fake_bridge and 0 <= tag <= 4095