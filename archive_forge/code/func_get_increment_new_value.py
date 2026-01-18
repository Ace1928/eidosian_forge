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
def get_increment_new_value(self):
    """Returns the final (incremented) value of the column in this
        transaction that was set to be incremented by Row.increment.  This
        transaction must have committed successfully."""
    assert self._status == Transaction.SUCCESS
    return self._inc_new_value