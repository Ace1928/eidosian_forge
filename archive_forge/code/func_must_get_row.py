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
def must_get_row(self, vsctl_table, record_id):
    ovsrec_row = self.get_row(vsctl_table, record_id)
    if not ovsrec_row:
        vsctl_fatal('no row "%s" in table %s' % (record_id, vsctl_table.table_name))
    return ovsrec_row