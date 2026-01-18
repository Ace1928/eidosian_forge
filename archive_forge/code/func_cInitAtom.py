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
def cInitAtom(self, var):
    if self.type == ovs.db.types.IntegerType:
        return '.integer = %d' % self.value
    elif self.type == ovs.db.types.RealType:
        return '.real = %.15g' % self.value
    elif self.type == ovs.db.types.BooleanType:
        if self.value:
            return '.boolean = true'
        else:
            return '.boolean = false'
    elif self.type == ovs.db.types.StringType:
        return '.s = %s' % escapeCString(self.value)
    elif self.type == ovs.db.types.UuidType:
        return '.uuid = %s' % ovs.ovsuuid.to_c_assignment(self.value)