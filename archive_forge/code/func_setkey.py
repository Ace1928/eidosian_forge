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
def setkey(self, column_name, key, value):
    self._idl.txn._txn_rows[self.uuid] = self
    column = self._table.columns[column_name]
    try:
        data.Datum.from_python(column.type, {key: value}, _row_to_uuid)
    except error.Error as e:
        vlog.err('attempting to write bad value to column %s (%s)' % (column_name, e))
        return
    if self._data and column_name in self._data:
        removes = self._mutations.setdefault('_removes', {})
        column_value = removes.setdefault(column_name, set())
        column_value.add(key)
    inserts = self._mutations.setdefault('_inserts', {})
    column_value = inserts.setdefault(column_name, {})
    column_value[key] = value