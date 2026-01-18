import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def has_constraints(self):
    return self.enum is not None or self.min is not None or self.max is not None or (self.min_length != 0) or (self.max_length != sys.maxsize) or (self.ref_table_name is not None)