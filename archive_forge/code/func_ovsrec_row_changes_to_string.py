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
def ovsrec_row_changes_to_string(ovsrec_row):
    if not ovsrec_row._changes:
        return ovsrec_row._changes
    return dict(((key, value.to_string()) for key, value in ovsrec_row._changes.items()))