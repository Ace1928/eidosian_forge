import collections
import enum
import functools
import uuid
import ovs.db.data as data
import ovs.db.parser
import ovs.db.schema
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.vlog
from ovs.db import custom_index
from ovs.db import error
def index_equal(self, table, name, value):
    """Return items in a named index matching a value"""
    return self.tables[table].rows.indexes[name].irange(value, value)