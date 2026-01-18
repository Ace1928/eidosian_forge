import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def copyCValue(self, dst, src, refTable=True):
    args = {'dst': dst, 'src': src}
    if self.ref_table_name:
        if not refTable:
            return '%(dst)s = *%(src)s;' % args
        return '%(dst)s = %(src)s->header_.uuid;' % args
    elif self.type == StringType:
        return '%(dst)s = ovsdb_atom_string_create(%(src)s);' % args
    else:
        return '%(dst)s = %(src)s;' % args