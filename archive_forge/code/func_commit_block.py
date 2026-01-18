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
def commit_block(self):
    """Attempts to commit this transaction, blocking until the commit
        either succeeds or fails.  Returns the final commit status, which may
        be any Transaction.* value other than Transaction.INCOMPLETE.

        This function calls Idl.run() on this transaction'ss IDL, so it may
        cause Idl.change_seqno to change."""
    while True:
        status = self.commit()
        if status != Transaction.INCOMPLETE:
            return status
        self.idl.run()
        poller = ovs.poller.Poller()
        self.idl.wait(poller)
        self.wait(poller)
        poller.block()