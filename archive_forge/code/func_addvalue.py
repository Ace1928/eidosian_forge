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
def addvalue(self, column_name, key):
    self._idl.txn._txn_rows[self.uuid] = self
    column = self._table.columns[column_name]
    try:
        data.Datum.from_python(column.type, key, _row_to_uuid)
    except error.Error as e:
        vlog.err('attempting to write bad value to column %s (%s)' % (column_name, e))
        return
    inserts = self._mutations.setdefault('_inserts', {})
    column_value = inserts.setdefault(column_name, set())
    column_value.add(key)