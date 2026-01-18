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
@staticmethod
def status_to_string(status):
    """Converts one of the status values that Transaction.commit() can
        return into a human-readable string.

        (The status values are in fact such strings already, so
        there's nothing to do.)"""
    return status