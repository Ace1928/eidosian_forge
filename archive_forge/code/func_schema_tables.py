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
@classmethod
def schema_tables(cls, idl, schema):
    return {k: cls(idl, v) for k, v in schema.tables.items()}