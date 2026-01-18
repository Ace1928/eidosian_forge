import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def is_weak_ref(self):
    return self.is_ref() and self.ref_type == 'weak'