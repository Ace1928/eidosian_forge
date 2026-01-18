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
def force_reconnect(self):
    """Forces the IDL to drop its connection to the database and reconnect.
        In the meantime, the contents of the IDL will not change."""
    if self.state == self.IDL_S_MONITORING:
        self._session.reset_backoff()
    self._session.force_reconnect()