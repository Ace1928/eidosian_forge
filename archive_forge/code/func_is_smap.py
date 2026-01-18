import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def is_smap(self):
    return self.is_map() and self.key.type == StringType and (self.value.type == StringType)