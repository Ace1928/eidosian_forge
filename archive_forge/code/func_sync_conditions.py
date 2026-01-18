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
def sync_conditions(self):
    """Synchronize condition state when the FSM is restarted

        If a non-zero last_id is available for the DB, then upon reconnect
        the IDL should first request acked conditions to avoid missing updates
        about records that were added before the transaction with
        txn-id == last_id. If there were requested condition changes in flight
        and the IDL client didn't set new conditions, then reset the requested
        conditions to new to trigger a follow-up monitor_cond_change request.

        If there were changes in flight then there are two cases:
        a. either the server already processed the requested monitor condition
           change but the FSM was restarted before the client was notified.
           In this case the client should clear its local cache because it's
           out of sync with the monitor view on the server side.

        b. OR the server hasn't processed the requested monitor condition
           change yet.

        As there's no easy way to differentiate between the two, and given that
        this condition should be rare, reset the 'last_id', essentially
        flushing the local cached DB contents.
        """
    ack_all = self.last_id == str(uuid.UUID(int=0))
    if ack_all:
        self.cond_changed = False
    for table in self.tables.values():
        if ack_all:
            table.condition_state.request()
            table.condition_state.ack()
        elif table.condition_state.reset():
            self.last_id = str(uuid.UUID(int=0))
            self.cond_changed = True