import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def default_atom(self):
    return ovs.db.data.Atom(self, self.default)