import functools
import re
import uuid
import ovs.db.parser
import ovs.db.types
import ovs.json
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.socket_util
from ovs.db import error
def conforms_to_type(self):
    n = len(self.values)
    return self.type.n_min <= n <= self.type.n_max