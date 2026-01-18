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
def restart_fsm(self):
    self.sync_conditions()
    self.__send_server_schema_request()
    self.state = self.IDL_S_SERVER_SCHEMA_REQUESTED