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
def has_ever_connected(self):
    """Returns True, if the IDL successfully connected to the remote
        database and retrieved its contents (even if the connection
        subsequently dropped and is in the process of reconnecting).  If so,
        then the IDL contains an atomic snapshot of the database's contents
        (but it might be arbitrarily old if the connection dropped).

        Returns False if the IDL has never connected or retrieved the
        database's contents.  If so, the IDL is empty."""
    return self.change_seqno != 0