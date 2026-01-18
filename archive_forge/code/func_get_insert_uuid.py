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
def get_insert_uuid(self, uuid):
    """Finds and returns the permanent UUID that the database assigned to a
        newly inserted row, given the UUID that Transaction.insert() assigned
        locally to that row.

        Returns None if 'uuid' is not a UUID assigned by Transaction.insert()
        or if it was assigned by that function and then deleted by Row.delete()
        within the same transaction.  (Rows that are inserted and then deleted
        within a single transaction are never sent to the database server, so
        it never assigns them a permanent UUID.)

        This transaction must have completed successfully."""
    assert self._status in (Transaction.SUCCESS, Transaction.UNCHANGED)
    inserted_row = self._inserted_rows.get(uuid)
    if inserted_row:
        return inserted_row.real
    return None