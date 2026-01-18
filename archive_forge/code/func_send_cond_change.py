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
def send_cond_change(self):
    if not self._session.is_connected() or self._request_id is not None:
        return
    msg = self.compose_cond_change()
    if msg:
        self.send_request(msg)